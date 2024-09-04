# -*- coding: utf-8 -*-
'''
@File    :   QuestionRe_writer.py
@Time    :   2024/09/02 10:00:37
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from LLM import llm

prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["question"],
)
question_rewriter = prompt | llm | StrOutputParser()
