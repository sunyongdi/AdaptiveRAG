# -*- coding: utf-8 -*-
'''
@File    :   RetrievalGrader.py
@Time    :   2024/08/30 17:41:02
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from LLM import llm

prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)
retrieval_grader = prompt | llm | JsonOutputParser()


