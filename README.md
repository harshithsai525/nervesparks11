
# Multi-Document Legal Research Assistant ⚖️

This project is a **Retrieval-Augmented Generation (RAG)** system built to analyze multiple legal documents and provide contextual answers to legal queries. It leverages a modern tech stack to deliver fast, accurate, and cited responses, making legal research more efficient.

## Features

  - **Multi-Document Analysis**: Upload and analyze various legal document formats like contracts, case law, and statutes.
  - **Contextual Answers**: Get answers based directly on the content of the provided documents.
  - **Source Citations**: The system is designed to cite the specific source document for the information it provides.
  - **Conflict Handling**: The model is prompted to identify and report conflicting information found across different sources.
  - **Fast & Efficient**: Powered by **Groq's** LPU Inference Engine for real-time responses and **Streamlit** for a clean user interface.

## System Architecture

The RAG system follows a logical data flow:

1.  **Load**: Legal documents (PDFs) are uploaded through the Streamlit interface.
2.  **Chunk**: The documents are broken down into smaller, semantically meaningful chunks.
3.  **Embed**: Each chunk is converted into a numerical vector (embedding) using a HuggingFace Sentence Transformer model.
4.  **Store**: The embeddings and their corresponding text chunks are stored in a **ChromaDB** vector database.
5.  **Retrieve**: When a user asks a query, the system embeds the query and retrieves the most relevant chunks from the vector database.
6.  **Generate**: The retrieved chunks (context) and the original query are passed to a **Groq** language model, which generates a comprehensive, context-aware answer with citations.

## Setup and Usage

Follow these steps to set up the project and run the application on your local machine.

### 1\. Create a Virtual Environment

A virtual environment is a self-contained directory that holds a specific Python interpreter and its own set of installed packages. This prevents package conflicts between projects.

Navigate to your project folder in your terminal and run the following command to create an environment named `legalenv`:

```bash
python -m venv legalenv
```

### 2\. Activate the Virtual Environment

Before you can install packages or run your app, you must **activate** the environment.

  * **On Windows (PowerShell/CMD):**

    ```powershell
    .\legalenv\Scripts\activate
    ```

  * **On macOS / Linux:**

    ```bash
    source legalenv/bin/activate
    ```

Your terminal prompt should now be prefixed with `(legalenv)`.

### 3\. Install Dependencies

With the environment active, install all the required libraries from the `requirements.txt` file with this single command:

```bash
pip install -r requirements.txt
```

### 4\. Run the Application

Now that your environment is set up and all dependencies are installed, you can launch the Streamlit application.

```bash
streamlit run app.py
```

This command starts a local web server, and your default browser will open with the running application. 🚀
