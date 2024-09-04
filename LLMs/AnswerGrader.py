# -*- coding: utf-8 -*-
'''
@File    :   AnswerGrader.py
@Time    :   2024/09/02 09:54:31
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from LLM import llm

prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)
answer_grader = prompt | llm | JsonOutputParser()


    
