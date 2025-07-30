
import os
import uuid
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain.memory import ConversationBufferMemory, BaseMemory
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import Dict, List, Any
from langchain.memory import ChatMessageHistory

# Define the database file path
db_file = "chat_memory/chat_history.db"
# Ensure the directory exists
os.makedirs(os.path.dirname(db_file), exist_ok=True)

# Create an SQLite engine
engine = create_engine(f"sqlite:///{db_file}")

# Define a base for declarative models
Base = declarative_base()

# Define the ChatMessage model
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    thread_id = Column(String)
    role = Column(String)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"ChatMessage(id={self.id}, role='{self.role}', content='{self.content[:50]}...')"

# Create the table in the database
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

class SQLChatMemory(BaseMemory):
    def __init__(self, memory_key="chat_history", thread_id=None):
        super().__init__()
        self.memory_key = memory_key
        self.thread_id = thread_id
        self.session = Session()
        self.embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chroma = Chroma(persist_directory="chat_memory/chroma", embedding_function=self.embed_model)
        self.message_history = ChatMessageHistory()

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Load messages from ChromaDB
        relevant_messages = self.get_relevant_messages(inputs["input"])
        # Load messages from short-term memory
        short_term_messages = self.message_history.messages
        messages = relevant_messages + short_term_messages
        return {self.memory_key: messages}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_message = inputs.get("input")
        output_message = outputs.get("output")
        if input_message:
            self.message_history.add_user_message(input_message)
            self.chroma.add_texts([input_message])
        if output_message:
            self.message_history.add_ai_message(output_message)
            self.chroma.add_texts([output_message])

    def clear(self):
        self.session.query(ChatMessage).filter_by(thread_id=str(self.thread_id)).delete()
        self.session.commit()
        self.chroma.delete_collection()

    def close(self):
        self.session.close()

    def get_relevant_messages(self, query):
        docs = self.chroma.similarity_search(query, k=3)
        messages = [HumanMessage(content=doc.page_content) for doc in docs]
        return messages

# Initialize the memory
memory = SQLChatMemory(memory_key="chat_history")
