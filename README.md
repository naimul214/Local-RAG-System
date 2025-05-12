# 🔍📄 Local Retrieval-Augmented Generation (RAG) System

A fully local Retrieval-Augmented Generation (RAG) application that lets you upload PDFs or web URLs and ask questions about their content. It combines text retrieval with language generation using Hugging Face models, FAISS for vector search, and is powered by a Streamlit frontend, FastAPI backend, and Docker Compose.

---

## 🚀 Features

- Upload a **PDF** (max ~10 pages) or a **web URL**
- Extract and chunk text using `PyPDF2` or `BeautifulSoup4`
- Generate **text embeddings** using `sentence-transformers/all-MiniLM-L6-v2`
- Store and retrieve vectors with **FAISS** for fast semantic search
- Generate human-like answers with **Hugging Face `t5-small`**
- Full app orchestration using **Docker Compose**
- No internet or cloud APIs required — everything runs locally!

---

## 📁 Project Structure

├── docker-compose.yml 
├── backend/ 
│ ├── Dockerfile 
│ ├── main.py 
│ └── requirements.txt 
└── frontend/
  ├── Dockerfile 
  ├── app.py 
  └── requirements.txt


---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/local-rag-app.git
cd local-rag-app

### 2. Build the Containers

docker compose build

### 3. Run the App

docker compose up


Streamlit (frontend): http://localhost:8501

FastAPI (backend): http://localhost:8000/docs

🧠 How It Works
Upload a PDF or URL from the Streamlit interface.

The FastAPI backend extracts and chunks the text.

Chunks are converted into semantic vectors using all-MiniLM-L6-v2.

Vectors are stored in a FAISS index for fast similarity search.

When a user asks a question:

The question is embedded as a vector

FAISS retrieves the most relevant chunks

Chunks + question are passed to t5-small to generate an answer

The Streamlit frontend displays the generated answer and (optionally) the source chunks.

📦 Models Used

Task	Model
Embedding	sentence-transformers/all-MiniLM-L6-v2
Generation	t5-small
🐳 Docker Notes
Backend and frontend are on the same Docker network (compose_default)

The frontend uses http://backend:8000 internally to talk to FastAPI

To run only one service (e.g., backend):

docker compose up backend


💡 Improvements & Ideas
Use pdfplumber for better PDF parsing accuracy

Switch to flan-t5-base or mistral-7B for more fluent responses (GPU required)

Add document history and session state tracking

Use Haystack or LangChain for modular pipeline configuration

📚 What I Learned
How FAISS can be used for vector-based search

How Hugging Face makes embedding and text generation easy

Importance of chunk size and text cleaning on retrieval quality

How to fully containerize and run an AI pipeline locally

🛠 Requirements (if running manually)
Python 3.9 or higher

pip install -r requirements.txt inside each service directory

Docker & Docker Compose (v2+ recommended)

🤝 Acknowledgments
Hugging Face Transformers

FAISS – Facebook AI Similarity Search

Streamlit

FastAPI

📜 License
MIT License – free to use and modify.

Developed for the NLP Assignment – April 2025
