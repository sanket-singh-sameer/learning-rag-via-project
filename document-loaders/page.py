from langchain_community.document_loaders import WebBaseLoader

url = "https://xsam.in"

loader = WebBaseLoader(url)
data = loader.load()

print(data)