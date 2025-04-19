import streamlit as st
import requests

st.set_page_config(page_title="Local RAG QA System", layout="wide")
st.title("📄🔍 Local RAG QA System")
st.write("Upload documents (PDFs or web pages) and ask questions. The system will retrieve relevant info from your documents and generate answers.")

# Backend API URL (assuming Docker Compose service name 'backend')
API_URL = "http://backend:8000"

# Session state to store history of Q&A and uploaded docs count
if "history" not in st.session_state:
    st.session_state.history = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = 0

# Upload Section
st.header("1. Add Document to Knowledge Base")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
url_input = st.text_input("...or enter a URL of a webpage")

if st.button("Add Document"):
    if uploaded_file:
        # Send PDF file to backend
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        try:
            res = requests.post(f"{API_URL}/upload", files=files)
            result = res.json()
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            result = None
    elif url_input:
        # Send URL to backend
        try:
            res = requests.post(f"{API_URL}/upload", data={"url": url_input})
            result = res.json()
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            result = None
    else:
        st.error("Please provide a PDF or URL to upload.")
        result = None

    if result:
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success(f"Document added! Chunks indexed: {result['chunks_added']} (Total chunks in knowledge base: {result['total_chunks']})")
            st.session_state.docs_loaded = result["total_chunks"]

# Query Section
st.header("2. Ask a Question")
question = st.text_input("Your question:", "")
if st.button("Get Answer"):
    if not question:
        st.error("Please enter a question.")
    elif st.session_state.docs_loaded == 0:
        st.error("No documents available. Please upload a document first.")
    else:
        # Send query to backend
        try:
            res = requests.post(f"{API_URL}/query", json={"question": question})
            answer_data = res.json()
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            answer_data = None

        if answer_data:
            if "error" in answer_data:
                st.error(f"Error: {answer_data['error']}")
            else:
                answer = answer_data.get("answer", "")
                top_chunks = answer_data.get("chunks", [])
                # Display the answer
                st.subheader("Answer:")
                st.write(answer)
                # Save Q&A to history
                st.session_state.history.append((question, answer))
                # Show the retrieved context chunks for transparency
                if top_chunks:
                    with st.expander("Retrieved Top Chunks", expanded=False):
                        for i, chunk in enumerate(top_chunks, start=1):
                            st.markdown(f"**Chunk {i}:** _{chunk}_")

# History Section (optional)
if st.session_state.history:
    st.header("3. Q&A History")
    for idx, (q, ans) in enumerate(st.session_state.history, start=1):
        st.markdown(f"**Q{idx}:** {q}")
        st.markdown(f"**A{idx}:** {ans}")
        st.markdown("---")