import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # Updated import
import fitz  # PyMuPDF

class LegalDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF using PyMuPDF"""
        try:
            docs = []
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text()
                    metadata = {
                        "source": os.path.basename(file_path),
                        "page_number": page_num,
                    }
                    docs.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))
            return self.text_splitter.split_documents(docs)
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []
    
    def process_directory(self, directory: str) -> List[Document]:
        """Process all PDFs in a directory"""
        all_docs = []
        for filename in os.listdir(directory):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory, filename)
                docs = self.process_pdf(file_path)
                all_docs.extend(docs)
        return all_docs