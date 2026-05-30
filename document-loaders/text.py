from langchain_community.document_loaders import TextLoader

loader = TextLoader("./document-loaders/text.txt")
doc = loader.load()
print(doc[0].page_content)