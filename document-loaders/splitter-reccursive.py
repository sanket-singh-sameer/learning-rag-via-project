from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader("./document-loaders/pdf.pdf")
doc = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

chunks = splitter.split_documents(doc)
print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()