from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


loader = TextLoader("./document-loaders/text.txt")
doc = loader.load()


splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

chunks = splitter.split_documents(doc)
print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()