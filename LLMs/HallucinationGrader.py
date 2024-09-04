# -*- coding: utf-8 -*-
'''
@File    :   HallucinationGrader.py
@Time    :   2024/08/30 17:54:08
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''

from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from LLM import llm


prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)
hallucination_grader = prompt | llm | JsonOutputParser()

    
