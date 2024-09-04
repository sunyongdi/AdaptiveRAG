## Adaptive RAG using local LLMs
本项目是对langchain 的Adaptive RAG教程进行整理，整体的代码比较简单，没有什么坑点。  
之前一直比较排斥langchain，觉得这种工具太过臃肿以及过度的封装，缺少了大模型灵活性，一直是原滋原味的手撸代码，这次用过之后有了很大的改观，工程做的非常的棒，以及一些逻辑非常值得学习，点赞。  
教程地址：https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/

![RAG](https://picgo-1305561115.cos.ap-beijing.myqcloud.com/img/20240904110554.png)

细节就不细说了，中途也没有太多的修改，其中的prompt没太大必要替换成中文，可以尝试，但我试的效果一般。  
## 参数配置
1. 创建config.py文件
```
MODEL_PATH = './model_hub/Qwen2-7B-Instruct'
EMBEDDING_PATH = './model_hub/bge-large-zh-v1___5'
EMBEDDING_MODEL = 'text2vec'
DEVICE = "cuda" # the device to load the model onto
TAVILY_API_KEY = 'xxxxxx'

IS_PROCESS = True
DOCS_PATH = './docs/laws'
DB_PATH = './vector/FAISS.db'
DB_INDEX = 'law_index'
```
2. 创建docs 文件夹
将txt文件或md放到文件内
3. 执行 python run.py
