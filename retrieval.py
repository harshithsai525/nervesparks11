from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from typing import List, Dict, Any
import os


class LegalRetriever:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )
        self.vector_db = None
        self.qa_chain = None

        self.prompt_template = """Answer based on legal context:
        {context}

        Question: {question}

        Provide detailed answer with citations:"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    def create_vector_db(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("No documents provided for vector DB creation.")

        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=None  # Use in-memory vectorstore
        )
        self._initialize_qa_chain()

    def _initialize_qa_chain(self) -> None:
        if not self.vector_db:
            raise RuntimeError("Vector store not initialized.")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                temperature=0,
                model_name="llama3-8b-8192",
                groq_api_key=os.getenv("GROQ_API_KEY")
            ),
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={
                "prompt": self.PROMPT,
                "document_variable_name": "context"
            },
            return_source_documents=True
        )

    def query(self, question: str) -> Dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if not self.qa_chain:
            raise RuntimeError("QA chain not initialized. Upload and process documents first.")

        result = self.qa_chain({"query": question})

        return {
            "answer": result["result"],
            "sources": [
                {
                    "document": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page_number", "N/A"),
                    "content": doc.page_content[:200] + "..."
                }
                for doc in result["source_documents"]
            ]
        }
