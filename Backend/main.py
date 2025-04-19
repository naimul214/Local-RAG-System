from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2, requests
from bs4 import BeautifulSoup

app = FastAPI(title="Local RAG QA Backend")

# Allow frontend (Streamlit) to call the API from another origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load embedding model and generative model once at startup
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# The MiniLM model outputs 384-dimensional sentence embeddings&#8203;:contentReference[oaicite:7]{index=7}
embedding_dim = 384

gen_tokenizer = AutoTokenizer.from_pretrained('t5-small')
gen_model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
# T5-small is a 60M parameter text-to-text model that can generate answers from context&#8203;:contentReference[oaicite:8]{index=8}

# Initialize FAISS index (cosine similarity via Inner Product). 
# Use IndexFlatIP and normalize embeddings for cosine similarity search.
index = faiss.IndexFlatIP(embedding_dim)
stored_chunks = []  # list to store text chunks corresponding to vectors

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20):
    """Split text into chunks of roughly `chunk_size` words, with `overlap` word overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
    return chunks

@app.post("/upload")
async def upload_document(file: UploadFile = File(None), url: str = Form(None)):
    """
    Ingest a document into the knowledge base, from a PDF file or a webpage URL.
    """
    # 1. Extract text from the input (PDF file or URL)
    raw_text = ""
    if file is not None:
        # Read PDF file content
        file_bytes = await file.read()
        try:
            reader = PyPDF2.PdfReader(file_bytes)
        except Exception as e:
            return {"error": f"Failed to read PDF: {e}"}
        # Extract text from each page of the PDF
        for page in reader.pages:
            if page.extract_text():
                raw_text += page.extract_text() + " "
    elif url:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            return {"error": f"Failed to fetch URL: {e}"}
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script and style elements, then get textual content
        for tag in soup(["script", "style"]):
            tag.decompose()
        raw_text = soup.get_text(separator=" ")
    else:
        return {"error": "No document provided. Please upload a file or provide a URL."}
    if not raw_text:
        return {"error": "No text could be extracted from the document."}

    # 2. Clean and chunk the text
    text = " ".join(raw_text.split())  # normalize whitespace
    chunks = chunk_text(text)
    if not chunks:
        return {"error": "Document is empty after processing."}

    # 3. Compute embeddings for each chunk and add to FAISS index
    new_vectors = embed_model.encode(chunks)            # shape: (num_chunks, 384)
    new_vectors = np.array([vec / np.linalg.norm(vec) for vec in new_vectors], dtype="float32")
    index.add(new_vectors)                              # add vectors to index
    stored_offset = len(stored_chunks)
    stored_chunks.extend(chunks)                        # store chunk texts

    return {"chunks_added": len(chunks), "total_chunks": len(stored_chunks)}

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_documents(req: QueryRequest):
    """
    Answer a user's question using the ingested document data.
    Finds relevant chunks via the FAISS index and generates an answer with T5.
    """
    question = req.question
    if not question:
        return {"error": "Question cannot be empty."}
    if index.ntotal == 0:
        return {"error": "No documents in knowledge base. Please upload a document first."}

    # 1. Embed the question and retrieve top matches from FAISS
    q_vec = embed_model.encode([question])
    q_vec = np.array([q_vec[0] / np.linalg.norm(q_vec[0])], dtype="float32")
    k = 3  # number of top chunks to retrieve
    distances, indices = index.search(q_vec, k)
    top_indices = indices[0]
    top_chunks = [stored_chunks[i] for i in top_indices if i < len(stored_chunks)]
    
    # 2. Prepare input for generative model (combine question with retrieved context)
    context = " ".join(top_chunks)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    # 3. Generate answer using T5-small
    input_ids = gen_tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    output_ids = gen_model.generate(input_ids, max_length=200, early_stopping=True)
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"answer": answer, "chunks": top_chunks}