### 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for efficient document processing and knowledge retrieval. It extracts text and tables from PDFs using the **Unstructured** library, stores raw PDFs in **Redis**, and indexes extracted embeddings in **PGVector** for semantic search. The system leverages **MultiVector Retriever** for context retrieval before querying an **LLM (GPT model)**.

![My Image](https://github.com/Mercytopsy/pdf-rag-chatbot-streamlit/blob/main/Architectural%20Diagram.png)


### 🚀 Features

- **Unstructured Document Processing**: Extracts text and tables from PDFs.  
- **Redis for Raw Storage**: Stores and retrieves raw PDFs efficiently, to implement persistent storage.  
- **PGVector for Vector Storage**: Indexes and retrieves high-dimensional embeddings for similarity search.  
- **MultiVector Retriever**: Optimized for retrieving contextual information from multiple sources.  
- **LLM Integration**: Uses a **GPT model** to generate responses based on retrieved context.  

### 🛠️ Tech Stack

#### Programming Language
- Python  

#### Libraries
- `unstructured`
- `pgvector`
- `redis`
- `langchain`
- `openai`

#### Databases
- **Redis**: For raw PDF storage  
- **PostgreSQL + PGVector**: For embeddings storage  

#### LLM
- **Gemini 2.5 Flash** (via Google Generative AI API)

---

### 🔄 Modifications

| Area | Original | Updated |
|------|----------|---------|
| LLM | `ChatOpenAI` (`gpt-4o-mini`) | `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) |
| Embeddings | `OpenAIEmbeddings` (`text-embedding-ada-002`) | `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`) |
| PDF strategy | `hi_res` (requires Tesseract) | `fast` (no system OCR dependency) |
| Database | `localhost:6024` (langchain user) | `localhost:5432` (postgres user) |
| Poppler | Not bundled | Bundled in `poppler/` folder, added to PATH at runtime |
| Dependencies removed | `nvidia-nccl-cu12`, `triton` (Linux-only) | Removed from `requirements.txt` |
