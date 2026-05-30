from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./document-loaders/pdf.pdf")
docs = loader.load()
print(len(docs))