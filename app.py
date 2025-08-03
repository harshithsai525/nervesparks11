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

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title & description
st.title("🧠 Legal Research Assistant")
st.markdown("""
This assistant analyzes legal documents (PDFs) and answers questions with citations.
Powered by Groq + HuggingFace + LangChain.
""")

# Sidebar: File Upload
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload legal PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )
    process_button = st.button("📚 Process Documents")

    if process_button and uploaded_files:
        with st.spinner("Processing..."):
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)

            for file in uploaded_files:
                path = os.path.join(temp_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())

            docs = document_processor.process_directory(temp_dir)

            # Store in session
            st.session_state.documents = docs
            st.session_state.vector_ready = False  # Mark for fresh embedding

            # Clean temp
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

            st.success(f"✅ {len(uploaded_files)} file(s) processed!")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    **Doc:** {src.get("document", "N/A")}  
                    **Page:** {src.get("page", "N/A")}  
                    **Excerpt:** {src.get("content", "")}
                    """)
                    st.divider()

# Handle chat input
if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_txt = ""

        try:
            # Build vector store if needed
            if not st.session_state.vector_ready:
                if not st.session_state.documents:
                    raise RuntimeError("❌ No documents loaded. Please upload PDFs.")
                retriever.create_vector_db(st.session_state.documents)
                st.session_state.vector_ready = True

            start = time.time()
            result = retriever.query(prompt)
            elapsed = time.time() - start

            answer = result.get("answer", "No answer.")
            sources = result.get("sources", [])
            answer += f"\n\n_Generated in {elapsed:.2f} sec_"

            for chunk in answer.split():
                response_txt += chunk + " "
                time.sleep(0.02)
                placeholder.markdown(response_txt + "▌")
            placeholder.markdown(response_txt)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_txt,
                "sources": sources
            })

            if sources:
                with st.expander("📚 Sources"):
                    for src in sources:
                        st.markdown(f"""
                        **Doc:** {src.get("document", "N/A")}  
                        **Page:** {src.get("page", "N/A")}  
                        **Excerpt:** {src.get("content", "")}
                        """)
                        st.divider()

        except Exception as e:
            st.error(f"⚠️ {str(e)}")
