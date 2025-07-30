from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from prompts.campus_waifu import prompt_template

from rag.retriever import vector_store
from chatbot.memory_manager import memory
import os
from dotenv import load_dotenv
load_dotenv()
#setting up our llm
api_key = os.getenv("QROQ_API_KEY")

llm = ChatGroq(
    model='deepseek-r1-distill-llama-70b',
    reasoning_format="hidden",
    api_key = api_key,
    max_tokens = 512
)

#prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)
#main chain
from langchain.chains import ConversationalRetrievalChain

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=True  # optional: for debugging
)