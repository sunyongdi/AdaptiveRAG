# -*- coding: utf-8 -*-
'''
@File    :   Generate.py
@Time    :   2024/08/30 17:43:39
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from LLM import llm

prompt = PromptTemplate(
    template="""human

    你是回答问题的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话，保持答案简洁，请用中文回答。

    Question: {question} 

    Context: {context} 


    Answer:""", 
    input_variables=["question", "context"]
)
# Chain
rag_chain = prompt | llm | StrOutputParser()