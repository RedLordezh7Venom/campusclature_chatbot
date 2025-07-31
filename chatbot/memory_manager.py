# Memory management for chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import os # Import os for path checking

chat_histories = {}
memory = ConversationBufferMemory(
    memory_key="chat_history",  # must match the key used in prompt
    return_messages=True        # so LLM sees individual messages
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rag.document_loader import pages
from rag.retriever import embeddings


# --- Modification starts here ---
PERSIST_DIRECTORY = "vector_memory"
if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
    # Load existing vector store if directory exists and is not empty
    vector_mem = Chroma(
        collection_name = "user_chats",
        embedding_function=embeddings,
        persist_directory = PERSIST_DIRECTORY
    )
    print(f"Loaded existing vector store from {PERSIST_DIRECTORY}")
else:
    # Create a new vector store if directory doesn't exist or is empty
    vector_mem = Chroma(
        collection_name = "user_chats",
        embedding_function=embeddings,
        persist_directory = PERSIST_DIRECTORY
    )
    print(f"Created new vector store at {PERSIST_DIRECTORY}")
# --- Modification ends here ---

mem_retriever = vector_mem.as_retriever(search_kwargs={"k":3})


def get_history(session_id: str) -> ChatMessageHistory:
        if session_id not in chat_histories:
            chat_histories[session_id] = ChatMessageHistory()
        return chat_histories[session_id]

def add_memory(text: str):
        vector_mem.add_texts([text]) 

def get_relevant_memory(query: str) -> str:
    results = mem_retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results]) if results else ""
