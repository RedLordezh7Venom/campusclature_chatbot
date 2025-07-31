import uuid
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
#prompt imports
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    # Removed MessagesPlaceholder
)

from langchain_openai import ChatOpenAI
from rag.retriever import vector_store
from prompts.campus_waifu import prompt_template
from prompts.campus_waifu import sys_prompt,chat_prompt

from chatbot.memory_manager import memory # Re-added memory import
import os
from dotenv import load_dotenv
load_dotenv()
#setting up our llm
api_key = os.getenv("GROQ_KEY")
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    max_tokens=512  # Add this line to stay within limits
)
system_message_prompt = SystemMessagePromptTemplate.from_template(sys_prompt)
human_message_prompt = HumanMessagePromptTemplate.from_template(chat_prompt)

#chat histories in ram
from chatbot.memory_manager import chat_histories,mem_retriever,get_history,get_relevant_memory,add_memory



prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt,
    # Removed MessagesPlaceholder(variable_name="chat_history")
])
#main chain
from langchain.chains import ConversationalRetrievalChain

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    memory=memory, # Re-added memory
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=True  # optional: for debugging
)
runnable = RunnableWithMessageHistory(
            qa_chain,
            get_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )

def generate_response(user_message: str, session_id: str = "default") -> str:
    # ✅ Retrieve relevant past memories for context
    past_context = get_relevant_memory(user_message)
    combined_input = f"{user_message}\n\nRelevant past memory:\n{past_context}" if past_context else user_message

    response = runnable.invoke(
        {"question": combined_input},
        config={"configurable": {"session_id": session_id}}
    )

    # ✅ Save to long-term memory for future use
    add_memory(f"User: {user_message}\nBot: {response['answer']}")

    return response['answer']

if __name__ == "__main__":
    q1 = "I have a science exam on sunday , and maths exam on monday"
    response = generate_response(q1, session_id="test_user_123")
    print(response)

    q2 = "What days do I have the exam??"
    response = generate_response(q2, session_id="test_user_123")
    print(response)

    # response = qa_chain.invoke("thursday ko mera  science  ka exam hai, wednesday ko maths ka exam hai ")
    # print(response['answer'])
    # print("==================================")
    # response = qa_chain.invoke("aaj kal mosam bada acha hai")
    # print(response['answer'])
    # print("==================================")
    # response = qa_chain.invoke("ek baar batana maine kya batai thi kon se tests hain konse days ko ??")
    # print(response['answer'])
    # print("==================================")
    # print(response)
