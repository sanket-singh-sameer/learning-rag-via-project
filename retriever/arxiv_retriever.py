from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True,
)

docs = retriever.invoke("large language model")


for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(f"Title: {doc.metadata['title']}")
    print(f"Authors: {', '.join(doc.metadata['authors'])}")
    print(f"Abstract: {doc.page_content}\n")