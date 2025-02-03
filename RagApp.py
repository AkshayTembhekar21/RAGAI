import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore  # Updated import
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your API keys
openai_api_key = os.environ.get("OPEN_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")


# Initialize Pinecone instance
pc = Pinecone(api_key=pinecone_api_key)

# Define Pinecone index name
index_name = "pdf-rag"

# Check if the index exists, else create one
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

index = pc.Index(index_name)

# Check if embeddings already exist in Pinecone
index_stats = index.describe_index_stats()
existing_vector_count = index_stats['total_vector_count']

def load_and_store_pdf():
    """Loads the PDF, splits it into chunks, and stores embeddings in Pinecone."""
    print("No existing embeddings found. Processing and storing data...")

    # Load PDF
    pdf_path = "./CFA.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Store in Pinecone
    vectorstore = PineconeVectorStore.from_documents(chunks, embedding_model, index_name=index_name)

    print(f"Stored {len(chunks)} chunks in Pinecone!")
    return vectorstore  # Return vectorstore

# Only process and store embeddings if no existing data is found
if existing_vector_count == 0:
    vectorstore = load_and_store_pdf()  # Store embeddings and return vectorstore
else:
    print("Embeddings already exist in Pinecone. Skipping insertion...")

    # Using the new PineconeVectorStore approach
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)

user_input = ""
while True:
    query = input('Your Prompt:\n')
    if query in ['quit', 'exit', 'bye']:
        print('Have a great Trip')
        break
    # Perform similarity search directly with the query string
    search_results = vectorstore.similarity_search(query, k=3)
    for idx, doc in enumerate(search_results):
        user_input += doc.page_content+"\n";
    print(user_input)
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=1)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    prompt = ChatPromptTemplate(
        input_variables=['context_query'],
        messages=[
            SystemMessage(content='''You are a chat Assistance. Answer the user's questions using the context below.
            {context_query}
            '''), 
            MessagesPlaceholder(variable_name='chat_history'),
            HumanMessagePromptTemplate.from_template('{context_query}')
        ]
    )
    combined_input = user_input + "\n" + query
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )
    response = chain.invoke({'context_query': combined_input})
    print(response['text'])
    print('-' * 50)

