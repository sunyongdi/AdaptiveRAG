# -*- coding: utf-8 -*-
'''
@File    :   FAISS.py
@Time    :   2024/09/02 10:14:03
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
import numpy as np
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, DirectoryLoader

from utils.texts_plitter import separate_list, write_check_file, ChineseTextSplitter

import warnings
warnings.filterwarnings('ignore')

class FAISSWrapper(FAISS):
    chunk_size = 250
    chunk_conent = True
    score_threshold = 0

    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 4, filter=None, fetch_k=None
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not self.chunk_conent:
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append(doc)
                continue
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, store_len - i)):
                break_flag = False
                for l in [i + k, i - k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break_flag = True
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                if break_flag:
                    break
        if not self.chunk_conent:
            return docs
        if len(id_set) == 0 and self.score_threshold > 0:
            return []
        id_list = sorted(list(id_set))
        id_lists = separate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = int(doc_score)
            docs.append((doc, doc_score))
        return docs
    

class VectorDB:
    def __init__(self, EMBEDDING_PATH:str, EMBEDDING_MODEL:str, DEVICE:str='cpu'):
        self.embedding_model_dict = {
                "text2vec": EMBEDDING_PATH,
            }   

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_dict[EMBEDDING_MODEL],model_kwargs={'device': DEVICE})

    def load_url(self, urls:List)->List:
        """读取网络链接的数据

        Args:
            urls (List): 网络连接

        Returns:
            List: 切分后的文本
        """
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)
        return doc_splits
    
    def load_txt(self, filepath:str)->List:
        """读取txt文本数据，PDF 会报错

        Args:
            filepath (str): txt 文件路径

        Returns:
            List: 切分后的文本
        """
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False)
        doc_splits = loader.load_and_split(textsplitter)
        write_check_file(filepath, doc_splits)
        return doc_splits
    
    def load_directory_loader(self, filepath:str)->List:
        """读取txt文本数据，PDF 会报错

        Args:
            filepath (str): txt 文件路径

        Returns:
            List: 切分后的文本
        """
        loader = DirectoryLoader(filepath, glob="**/*.md", loader_cls=TextLoader, show_progress=True, use_multithreading=True)
        # loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False)
        doc_splits = loader.load_and_split(textsplitter)
        write_check_file(filepath, doc_splits)
        return doc_splits

    def get_retriever(self, doc_splits: List, save_local=True, folder_path:str='./vector/FAISS.db', index_name:str='law-index'):
        """获取retriever

        Args:
            doc_splits (List): 文档列表
            save_local (bool, optional): 是否保存到本地. Defaults to True.
            folder_path (str, optional): 保存路径. Defaults to './vector/FAISS.db'.
            index_name (str, optional): 索引名称. Defaults to 'law-index'.

        Returns:
            _type_: _description_
        """
        vectorstore = FAISSWrapper.from_documents(doc_splits, self.embeddings)
        if save_local:
            vectorstore.save_local(folder_path='./vector/FAISS.db', index_name='law-index')
        retriever = vectorstore.as_retriever()
        return retriever
    
    def load_local(self, folder_path:str='./vector/FAISS.db', index_name:str='law-index'):
        """加载本地向量库

        Returns:
            _type_: retriever
        """
        vectorstore = FAISSWrapper.load_local(folder_path=folder_path, embeddings=self.embeddings, index_name=index_name, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        return retriever

    

if __name__ == '__main__':
        from config import EMBEDDING_PATH, EMBEDDING_MODEL, DEVICE
        vector_db = VectorDB(EMBEDDING_PATH, EMBEDDING_MODEL, DEVICE)
        urls = [
            # "https://lilianweng.github.io/posts/2023-06-23-agent/",
            # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
            "https://www.flfgk.com/detail/83618d707d052bd24955814c12c20d66.html",
        ]
        doc_splits = vector_db.load_url(urls)
        retriever = vector_db.get_retriever(doc_splits, save_local=False)
        docs = retriever.get_relevant_documents('什么是倾销与损害')
        print(docs[1].page_content)
        # doc_splits = vector_db.load_txt(filepath='/root/sunyd/llms/版面分析/docs/甘肃祁连山国家级自然保护区管理条例2017.pdf')
        # retriever = vector_db.get_retriever(doc_splits)
        # docs = retriever.get_relevant_documents('该保护区禁止什么？')
        # print(docs[1].page_content)

