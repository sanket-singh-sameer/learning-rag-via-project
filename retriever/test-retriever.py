from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(
        page_content="Python is a programming language widely used for machine learning, automation, and web development.",
        metadata={"source": "doc1", "topic": "python"}
    ),
    Document(
        page_content="Many developers use Python for AI applications because of libraries such as NumPy, Pandas, and TensorFlow.",
        metadata={"source": "doc2", "topic": "python"}
    ),
    Document(
        page_content="Pandas is a Python library used for data analysis and manipulation.",
        metadata={"source": "doc3", "topic": "python"}
    ),
    Document(
        page_content="The Pandas package has become extremely popular among data scientists for handling tabular data.",
        metadata={"source": "doc4", "topic": "python"}
    ),
    Document(
        page_content="The Pandas package has become extremely popular among data scientists for handling tabular data.",
        metadata={"source": "doc5", "topic": "python"}
    ),
    Document(
        page_content="Machine learning enables computers to learn patterns from data without being explicitly programmed.",
        metadata={"source": "doc6", "topic": "ml"}
    ),
    Document(
        page_content="Deep learning is a subset of machine learning that uses neural networks with many layers.",
        metadata={"source": "doc7", "topic": "ml"}
    ),
    Document(
        page_content="PostgreSQL is a powerful open-source relational database management system.",
        metadata={"source": "doc8", "topic": "database"}
    ),
    Document(
        page_content="Prisma is an ORM that simplifies database access in Node.js and TypeScript applications.",
        metadata={"source": "doc9", "topic": "database"}
    ),
    Document(
        page_content="Technology has transformed the way people communicate, learn, and work.",
        metadata={"source": "doc10", "topic": "technology"}
    ),
    Document(
        page_content="Modern communication heavily relies on digital technologies and internet connectivity.",
        metadata={"source": "doc11", "topic": "technology"}
    ),
    Document(
        page_content="Climate change is one of the most pressing global challenges of our time.",
        metadata={"source": "doc12", "topic": "climate"}
    ),
    Document(
        page_content="Rising global temperatures contribute to extreme weather events and sea-level rise.",
        metadata={"source": "doc13", "topic": "climate"}
    ),
    Document(
        page_content="Python is also the name of a large non-venomous snake found in Africa and Asia.",
        metadata={"source": "doc14", "topic": "animal"}
    ),
    Document(
        page_content="Artificial intelligence systems are increasingly used in healthcare, finance, and education.",
        metadata={"source": "doc15", "topic": "ai"}
    ),
    Document(
        page_content="AI applications are becoming common in hospitals, banks, and schools.",
        metadata={"source": "doc16", "topic": "ai"}
    ),
]

embeddings_model = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    persist_directory="./retriever/test-retriever-chroma-db",
)

query = "artificial intelligence in hospitals"


print()
print()
print("----- Similarity Search Results -----")

similarity_search_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

results = similarity_search_retriever.invoke(query)

for result in results:
    print(f"Source: {result.metadata['source']}")
    print(f"Topic: {result.metadata['topic']}")
    print(f"Content: {result.page_content}\n")
    
print("----- End of Similarity Search Results -----")
print()
print()

print("----- MMR Search Results -----")

mmr_search_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2}
)

results = mmr_search_retriever.invoke(query)

for result in results:
    print(f"Source: {result.metadata['source']}")
    print(f"Topic: {result.metadata['topic']}")
    print(f"Content: {result.page_content}\n")
    
print("----- End of MMR Search Results -----")
print()
print()


print("----- MultiQuery Search Results -----")

retriever = vector_store.as_retriever()
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.7)
multiquery_search_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)
results = multiquery_search_retriever.invoke(query)

for result in results:
    print(f"Source: {result.metadata['source']}")
    print(f"Topic: {result.metadata['topic']}")
    print(f"Content: {result.page_content}\n")
    
print("----- End of MultiQuery Search Results -----")
print()
print()

