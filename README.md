# RAGAI  
AI RAG Application  

## Installation  

After cloning the repo, run the following command to install the required dependencies:  

```sh
pip install -r requirements.txt
(This requires pip and Python to be installed locally.)

##Usage
1️⃣ Add a PDF named CFA.pdf in the working directory.
2️⃣ Run the following command:
python RagApp.py
##How It Works
When a user asks a question based on the content in the PDF, Pinecone performs a similarity search using embeddings and fetches the top 3 results.
These 3 results (visible before the AI response) are passed to OpenAI’s API.
OpenAI then generates an answer based on the retrieved context.
