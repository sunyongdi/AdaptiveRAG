import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EMBEDDING_PATH, EMBEDDING_MODEL, DEVICE, DOCS_PATH, IS_PROCESS, DB_PATH, DB_INDEX
from FAISS import VectorDB
vector_db = VectorDB(EMBEDDING_PATH, EMBEDDING_MODEL, DEVICE)
if IS_PROCESS:
    doc_splits = vector_db.load_directory_loader(DOCS_PATH)
    retriever = vector_db.get_retriever(doc_splits)
else:
    retriever = vector_db.load_local(folder_path=DB_PATH, index_name=DB_INDEX)
