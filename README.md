### 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for efficient document processing and knowledge retrieval. It extracts text and tables from PDFs using the **Unstructured** library, stores raw document chunks in **Redis**, and indexes extracted embeddings in **PGVector** for semantic search. The system leverages **MultiVector Retriever** for context retrieval before querying **Gemini 2.5 Flash**.

![My Image](https://github.com/Mercytopsy/pdf-rag-chatbot-streamlit/blob/main/Architectural%20Diagram.png)


### 🚀 Features

- **Unstructured Document Processing**: Extracts text and tables from PDFs.
- **Redis for Raw Storage**: Stores and retrieves raw document chunks efficiently for persistent storage.
- **PGVector for Vector Storage**: Indexes and retrieves high-dimensional embeddings for similarity search.
- **MultiVector Retriever**: Optimized for retrieving contextual information from multiple sources.
- **Gemini Integration**: Uses **Gemini 2.5 Flash** for summarization and RAG responses, and **gemini-embedding-001** for embeddings.
- **Incremental Upload Progress**: Live step-by-step progress bar in the sidebar during PDF processing.
- **Uploaded PDF List**: Sidebar displays all previously processed PDFs.
- **Clear All Data**: One-click button to wipe Redis and PGVector for a fresh start.

### 🛠️ Tech Stack

#### Programming Language
- Python

#### Libraries
- `unstructured`
- `langchain-postgres` (PGVector)
- `langchain-google-genai`
- `redis`
- `langchain`
- `streamlit`

#### Databases
- **Redis**: Raw document chunk storage (keyed by `doc_id`)
- **PostgreSQL + PGVector**: Embedding vectors + LLM summaries

#### LLM & Embeddings
- **LLM**: `gemini-2.5-flash` (via Google Generative AI API)
- **Embeddings**: `gemini-embedding-001`

### ⚙️ Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_google_api_key
   PG_USER=postgres
   PG_PASSWORD=your_password
   PG_HOST=localhost
   PG_PORT=5432
   PG_DATABASE=postgres
   ```

3. Start Redis and PostgreSQL, then run:
   ```bash
   streamlit run RAG_with_streamlit.py
   ```

---

### 🔄 Modifications from Original

| Area | Original | Updated |
|------|----------|---------|
| LLM | `ChatOpenAI` (`gpt-4o-mini`) | `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) |
| Embeddings | `OpenAIEmbeddings` | `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`) |
| PDF strategy | `hi_res` (requires Tesseract) | `fast` (no system OCR dependency) |
| Database | `localhost:6024` (langchain user) | `localhost:5432` (configurable via `.env`) |
| Poppler | Not bundled | Bundled in `poppler/` folder, added to PATH at runtime |
| Upload UX | Silent processing on first chat | Incremental progress bar on upload |
| PDF history | Not shown | Listed in sidebar |
| Data management | No reset | Clear All Data button wipes Redis + PGVector |
| Dependencies removed | `nvidia-nccl-cu12`, `triton` (Linux-only) | Removed from `requirements.txt` |
