import uuid
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from prompts.campus_waifu import prompt_template

from rag.retriever import vector_store
from chatbot.memory_manager import memory, SQLChatMemory
import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

# Thread ID = user/session
thread_id = uuid.uuid4()

#setting up our llm
api_key = os.getenv("OPENROUTER_KEY")
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="openai/gpt-4o",
    max_tokens=512  # Add this line to stay within limits
)

# Initialize the memory
memory = SQLChatMemory(memory_key="chat_history", thread_id=thread_id)
message_history = ChatMessageHistory()

# Prompt
system_prompt = """You are a helpful assistant that answers questions about a document.
You are given the following extracted parts of a long document and a question.
Answer the question based on the context provided.
You should generate natural language responses."""

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}")
])

# RAG components
vectorstore = Chroma(persist_directory="chat_memory/chroma", embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 1. Create a prompt template for history-aware retrieval
template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

# 2. Create a history-aware retriever
def get_history_aware_retriever(llm, retriever, prompt):
    return create_history_aware_retriever(llm, retriever, prompt)

history_aware_retriever = get_history_aware_retriever(llm, retriever, CONDENSE_QUESTION_PROMPT)

# 3. Create a prompt template for combine docs
combine_docs_template = """Given the following extracted parts of a long document and a question, answer the question based on the context provided.
{context}

Question: {input}"""
COMBINE_DOCS_PROMPT = PromptTemplate.from_template(combine_docs_template)

# 4. Create a combine documents chain
def get_combine_documents_chain(llm, prompt):
    return create_stuff_documents_chain(llm, prompt)

combine_documents_chain = get_combine_documents_chain(llm, COMBINE_DOCS_PROMPT)

# 5. Create a retrieval chain
def get_retrieval_chain(retriever, combine_documents_chain):
    return create_retrieval_chain(retriever, combine_documents_chain)

retrieval_chain = get_retrieval_chain(history_aware_retriever, combine_documents_chain)

# 6. Create a chain with message history
def get_chain_with_message_history(prompt, llm, retrieval_chain, memory_key="chat_history"):
    return prompt | llm | memory

chain = get_chain_with_message_history(prompt, llm, retrieval_chain)

if __name__ == "__main__":
    response = chain.invoke({"input": "thursday ko mera  science  ka exam hai, wednesday ko maths ka exam hai "})
    print(response)
    print("==================================")
    response = chain.invoke({"input": "aaj kal mosam bada acha hai"})
    print(response)
    print("==================================")
    response = chain.invoke({"input": "ek baar batana maine kya batai thi kon se tests hain konse days ko ??"})
    print(response)
    print("==================================")
