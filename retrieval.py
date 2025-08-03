from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from typing import List, Dict, Any
import os

class LegalRetriever:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.persist_directory = persist_directory
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

        # Try to load vector DB only if it exists
        self._auto_initialize()

    def _auto_initialize(self) -> bool:
        if os.path.exists(self.persist_directory):
            try:
                self.vector_db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model
                )
                return True
            except Exception as e:
                print(f"[Warning] Failed to load existing vector DB: {e}")
        return False

    def create_vector_db(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("No documents provided")
        
        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        self.vector_db.persist()
        self._initialize_qa_chain()

    def _initialize_qa_chain(self) -> None:
        if not self.vector_db:
            raise RuntimeError("Vector DB not initialized")

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
        
        if not self.vector_db:
            raise RuntimeError("No documents loaded. Upload a PDF and initialize the vector DB.")
        
        if not self.qa_chain:
            self._initialize_qa_chain()

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
