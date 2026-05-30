from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("./test-files/dl-book.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that summerizes the document or the content inside it."),
    ("human", "{data}"),
])

final_prompt = template.format_messages(data=chunks)
print(len(chunks))
# response = model.invoke(final_prompt)
# print(response.content)