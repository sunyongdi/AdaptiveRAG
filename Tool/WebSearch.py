# -*- coding: utf-8 -*-
'''
@File    :   WebSearch.py
@Time    :   2024/09/02 15:17:08
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)