# RAGAI
AI RAG Application
After Cloning the repo run the following command to install the required dependencies
  pip install -r requirements.txt
(This will require to have pip and python installed on local)
Add a pdf named CFA.pdf in the working directory.
As last step run
python .\RagApp.py
When user will ask a question based on the content in pdf, the pinecone will do a similarity search based on embeddings and will fetch top 3 results(Youl'll see these results before the response from the AI)
These 3 results along with the question will be passed on to the openAI api and openAI api will create and ans based on it.
