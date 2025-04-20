FROM python:3.10-slim-bullseye

# Working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models to speed up runtime (cache in image)
RUN python -c "\
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
AutoModelForSeq2SeqLM.from_pretrained('t5-small'); \
AutoTokenizer.from_pretrained('t5-small'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
import faiss" 

# Copy application code
COPY main.py .

# Expose port and run the FastAPI app with Uvicorn
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]