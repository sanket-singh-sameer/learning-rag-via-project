# load pdf
# split into chunks
# create embeddings
# store in vector database

import time

from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_mistralai import MistralAIEmbeddings
# from langchain_ollama  import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

start = time.time()
print(start)

embeddings_model = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3",
)


document_loader = PyPDFLoader("./test-files/dl-book.pdf")
docs = document_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="./src/chroma-db-huggingface-endpoint",
)

end = time.time()
print(end)

print(f"Time taken: {end - start} seconds")