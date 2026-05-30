import time

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

embedding_model = MistralAIEmbeddings()
llm_model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

vector_store = Chroma(
    persist_directory="./src/chroma-db-mistral",
    embedding_function=embedding_model,
)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    }
)


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant AI Assistant that provides information only from the provided context. If the answer is not present in the context, say 'I don't know or i am unable to find the information from the book.'."),
        ("human", "Context: {context} & Question: {question}"),
    ]
)


print(" Press \"exit\" to quit the program.")
print("----- I know Everything About The Book \"Deep Learning by O'Reilly\"-----")


while True:
    user_prompt = input("You: ")
    start_time = time.time()
    if user_prompt.lower() == "exit":
        print("Exiting the program. Goodbye!")
        break
    
    mmr_results = retriever.invoke(user_prompt)
    context = "\n\n".join([f"{result.page_content}" for result in mmr_results])
    
    final_prompt = prompt_template.format_messages(
        context=context,
        question=user_prompt,
    )

    response = llm_model.invoke(final_prompt)
    end_time = time.time()
    print(f"AI: {response.content}\n")
    print(f"Response Time: {end_time - start_time:.2f} seconds\n")