# -*- coding: utf-8 -*-
'''
@File    :   Router.py
@Time    :   2024/08/30 17:39:32
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
from typing import Dict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from LLM import llm


prompt = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore or web search. \n
    Use the vectorstore for questions on legal. \n
    You do not need to be stringent with the keywords in the question related to these topics. \n
    Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
    Return the a JSON with a single key 'datasource' and no premable or explanation. \n
    Question to route: {question}""",
    input_variables=["question"],
)

question_router = prompt | llm | JsonOutputParser()

