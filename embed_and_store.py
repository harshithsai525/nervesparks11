from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_vectorstore():
    embedding = HuggingFaceEmbeddings()
    return Chroma(persist_directory="vectorstore", embedding_function=embedding)
