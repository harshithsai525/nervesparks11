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
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            documents = document_processor.process_directory(temp_dir)
            retriever.create_vector_db(documents)

            # Cleanup
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

            st.success(f"Processed {len(uploaded_files)} documents with {len(documents)} chunks!")

# Chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources and References"):
                for source in message["sources"]:
                    st.markdown(f"""
                    **Document:** {source.get('document', 'N/A')}  
                    **Page:** {source.get('page', 'N/A')}  
                    **Content:** {source.get('content', '')}  
                    """)
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            start_time = time.time()
            result = retriever.query(prompt)
            response_time = time.time() - start_time

            # Extract response and sources
            answer = result.get("answer", "No answer generated.")
            sources = result.get("sources", [])

            answer += f"\n\n_Generated in {response_time:.2f} seconds_"

            for chunk in answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })

            if sources:
                with st.expander("Sources and References"):
                    for source in sources:
                        st.markdown(f"""
                        **Document:** {source.get('document', 'N/A')}  
                        **Page:** {source.get('page', 'N/A')}  
                        **Content:** {source.get('content', '')}  
                        """)
                        st.divider()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
