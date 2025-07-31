from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///memory/chat_history.db"
    )
