import os
import streamlit as st
import requests

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="Enterprise Hybrid RAG", page_icon="📚", layout="wide")

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://enterprise-hybrid-rag.onrender.com"
).strip().rstrip("/")
API_PREFIX = BACKEND_URL
backend_status = "configured"

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("📚 Enterprise Hybrid RAG")
    st.markdown("---")

    try:
        health = requests.get(f"{API_PREFIX}/api/v1/health", timeout=5)
        if health.status_code == 200:
            st.success("Backend Online")
        else:
            st.error(f"Backend Error: {health.status_code}")
            st.code(health.text)
    except Exception as e:
        st.error(f"Backend Offline: {e}")

    if API_PREFIX is not None:
        try:
            health = requests.get(f"{API_PREFIX}/api/v1/health", timeout=5)
            if health.status_code == 200:
                st.success("Backend Online")
            else:
                st.error(f"Backend Error: {health.status_code}")
                st.code(health.text)
        except Exception as e:
            st.error(f"Backend Offline: {e}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# MAIN INTERFACE
# =========================
st.title("📚 Enterprise Hybrid RAG")

# 1. Upload Section
st.subheader("📤 Upload Documents")
uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file and st.button("Upload PDF"):
    with st.spinner("Uploading and processing document..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_PREFIX}/api/v1/upload", files=files, timeout=600)

            if response.status_code in [200, 201]:
                st.success(f"Successfully uploaded {uploaded_file.name}")
                if uploaded_file.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(uploaded_file.name)
            else:
                st.error(f"Upload Failed (Status {response.status_code})")
                st.code(response.text) # Displaying the specific server error
        except Exception as e:
            st.error(f"Connection Error: {e}")

# 2. Query Section
st.subheader("💬 Ask Questions")
query = st.text_input("Enter your question")

if st.button("Ask") and query:
    with st.spinner("Retrieving answer..."):
        payload = {"query": query, "top_k": 5, "stream": False}
        try:
            response = requests.post(
                f"{API_PREFIX}/api/v1/ask", json=payload, timeout=300
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.messages.append({"question": query, "response": data})
            else:
                st.error(f"Server Error ({response.status_code})")
                st.code(response.text) # Displaying the specific backend error
        except Exception as e:
            st.error(f"Connection Error: {e}")

# 3. Chat History
for item in reversed(st.session_state.messages):
    st.markdown("---")
    st.markdown(f"### 🙋 Question\n{item['question']}")
    data = item["response"]
    st.markdown("### 🤖 Answer")
    st.write(data.get("answer", "No answer returned"))
