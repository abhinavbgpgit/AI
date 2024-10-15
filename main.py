from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Load the FAISS index from the saved directory
embeddings = OllamaEmbeddings(model="llama2")
db = FAISS.load_local("faiss_index", embeddings)

# Run the query on the loaded vector store
query = "What are JavaScript Reserved Words?"
result = db.similarity_search(query)

print(result[0].page_content)