from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="Technology has transformed the way people communicate, learn, and work.", metadata={"source": "doc1"}),
    Document(page_content="Pandas is used for data analysis in python.", metadata={"source": "doc2"}),
    Document(page_content="Climate change is one of the most pressing issues of our time.", metadata={"source": "doc3"}),
    Document(page_content="a very famous package named Pandas in python is gettingb popular nowadays.", metadata={"source": "doc4"}),
    Document(page_content="Technology has transformed the way people communicate, learn, and work.", metadata={"source": "doc5"}),
]

embedding_model = HuggingFaceEmbeddings()


vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./vector-store/chroma-db",
)


result = vector_store.similarity_search("What is pandas used for?", k=2)

for r in result:
    print(r)
