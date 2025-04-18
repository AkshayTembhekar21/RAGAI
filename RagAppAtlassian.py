import os
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.environ.get("OPEN_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
confluence_email = os.environ.get("CONFLUENCE_EMAIL")  # Your Atlassian email
confluence_api_token = os.environ.get("CONFLUENCE_API_TOKEN")  # API token

confluence_base_url = "enter your link"
page_id = "page id from url"

pc = Pinecone(api_key=pinecone_api_key)
index_name = "pdf-rag"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

index_stats = index.describe_index_stats()
existing_vector_count = index_stats['total_vector_count']


def fetch_confluence_content(page_id):
    url = f"{confluence_base_url}/{page_id}?expand=body.storage"
    headers = {"Accept": "application/json"}
    
    response = requests.get(url, headers=headers, auth=(confluence_email, confluence_api_token))
    
    if response.status_code == 200:
        page_data = response.json()
        html_content = page_data["body"]["storage"]["value"]
        return html_content
    else:
        print(f"Failed to fetch page content: {response.status_code}")
        return None


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ").strip()


def load_and_store_confluence():
    print("Fetching Confluence content...")
    html_content = fetch_confluence_content(page_id)

    if not html_content:
        print("No content fetched. Exiting.")
        return None

    clean_text = clean_html(html_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(clean_text)

    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectorstore = PineconeVectorStore.from_texts(chunks, embedding_model, index_name=index_name)

    print(f"Stored {len(chunks)} chunks in Pinecone!")
    return vectorstore

if existing_vector_count == 0:
    vectorstore = load_and_store_confluence()
else:
    print("Embeddings already exist in Pinecone. Skipping insertion...")

    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)


while True:
    query = input('Your Prompt:\n')
    if query.lower() in ['quit', 'exit', 'bye']:
        print('Goodbye!')
        break

    search_results = vectorstore.similarity_search(query, k=3)
    retrieved_text = "\n".join([doc.page_content for doc in search_results])

    print("\n🔍 Top Matching Content:\n", retrieved_text)

    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0.7)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    prompt = ChatPromptTemplate(
        input_variables=['context_query'],
        messages=[
            SystemMessage(content="You are a helpful AI assistant. Answer the user's question using the provided context:\n{context_query}"),
            MessagesPlaceholder(variable_name='chat_history'),
            HumanMessagePromptTemplate.from_template("{context_query}")
        ]
    )

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
    response = chain.invoke({'context_query': retrieved_text + "\n" + query})

    print("\n🤖 AI Response:\n", response['text'])
    print('-' * 50)
