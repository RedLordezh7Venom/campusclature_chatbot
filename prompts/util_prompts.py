from langchain_core.prompts import PromptTemplate

condense_question_prompt = PromptTemplate.from_template("""
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone query.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:""")
