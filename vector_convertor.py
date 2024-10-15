from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Load the PDF file
loader = PyPDFLoader("javascript.pdf")

# Process and load documents
docs = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# Create embeddings
embeddings = OllamaEmbeddings(model="llama2")

# Create a FAISS vector store and save it to disk
db = FAISS.from_documents(documents[:2], embeddings)
# db.save_local("faiss_index")  # Save the index locally

query = "What are JavaScript Reserved Words?"
result = db.similarity_search(query)

print(result[0].page_content)
