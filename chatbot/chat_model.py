import uuid
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_openai import ChatOpenAI
from rag.retriever import vector_store
from prompts.campus_waifu import prompt_template
from prompts.campus_waifu import sys_prompt,chat_prompt

from chatbot.memory_manager import memory
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

prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])
#main chain
from langchain.chains import ConversationalRetrievalChain

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=True  # optional: for debugging
)

if __name__ == "__main__":
    response = qa_chain.invoke("thursday ko mera  science  ka exam hai, wednesday ko maths ka exam hai ")
    print(response['answer'])
    print("==================================")
    response = qa_chain.invoke("aaj kal mosam bada acha hai")
    print(response['answer'])
    print("==================================")
    response = qa_chain.invoke("ek baar batana maine kya batai thi kon se tests hain konse days ko ??")
    print(response['answer'])
    print("==================================")
    print(response)