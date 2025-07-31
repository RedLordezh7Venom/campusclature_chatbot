# Memory management for chat history
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",  # must match the key used in prompt
    return_messages=True,
    output_key = "answer"        
)