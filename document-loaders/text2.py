# this breaks the doc line by line and save it it the list

from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader("./document-loaders/text.txt")
docs = loader.load()
print(docs)