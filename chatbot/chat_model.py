from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from prompts.campus_waifu import prompt_template

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

from langchain.chains import ConversationalRetrievalChain

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=True  # optional: for debugging
)