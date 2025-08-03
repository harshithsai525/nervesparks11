import streamlit as st
from document_processor import LegalDocumentProcessor
from retrieval import LegalRetriever
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize components
document_processor = LegalDocumentProcessor()
retriever = LegalRetriever()

# App title and description
st.title("Multi-Document Legal Research Assistant")
st.markdown("""
This system analyzes multiple legal documents (contracts, case law, statutes) and provides 
contextual answers with proper citations. Upload your documents and ask legal questions.
Powered by Groq's ultra-fast Mixtral model.
""")

# Sidebar for document upload
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload legal documents (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    process_button = st.button("Process Documents")
    
    if process_button and uploaded_files:
        with st.spinner("Processing documents..."):
            # Save uploaded files temporarily
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Process documents
            documents = document_processor.process_directory(temp_dir)
            
            # Create vector database
            retriever.create_vector_db(documents)
            
            # Clean up temp files
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
            
            st.success(f"Processed {len(uploaded_files)} documents with {len(documents)} chunks!")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources and References"):
                for source in message["sources"]:
                    st.markdown(f"""
                    **Document:** {source['document']}  
                    **Page:** {source['page']}  
                    **Content:** {source['content']}  
                    """)
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a legal question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Get response from retriever
            start_time = time.time()
            result = retriever.query(prompt)
            response_time = time.time() - start_time
            
            answer = result["answer"]
            sources = result["sources"]
            
            # Add response time to answer
            answer = f"{answer}\n\n_Generated in {response_time:.2f} seconds_"
            
            # Simulate streaming
            for chunk in answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": sources
            })
            
            # Show sources
            with st.expander("Sources and References"):
                for source in sources:
                    st.markdown(f"""
                    **Document:** {source['document']}  
                    **Page:** {source['page']}  
                    **Content:** {source['content']}  
                    """)
                    st.divider()
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")