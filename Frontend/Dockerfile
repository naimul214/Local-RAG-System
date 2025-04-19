FROM python:3.10-slim-bullseye

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501
# Streamlit will run on port 8501 by default
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]