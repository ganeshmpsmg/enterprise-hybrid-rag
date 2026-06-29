import os
import streamlit as st
import requests

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="Enterprise Hybrid RAG", page_icon="📚", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "").strip().rstrip("/")
BACKEND_LOCAL_FALLBACK = os.getenv("BACKEND_LOCAL_FALLBACK", "false").lower() in ("1", "true", "yes")
if BACKEND_URL:
    API_PREFIX = BACKEND_URL
    backend_status = "configured"
elif BACKEND_LOCAL_FALLBACK:
    API_PREFIX = "http://localhost:8000"
    backend_status = "local_fallback"
else:
    API_PREFIX = None
    backend_status = "unset"

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

    # System Status
    if backend_status == "unset":
        st.error(
            "BACKEND_URL is not configured. Set BACKEND_URL to the deployed backend host, "
            "for example https://my-backend.example.com. The Streamlit app cannot connect until this is configured."
        )
    elif backend_status == "local_fallback":
        st.warning(
            "Using http://localhost:8000 as a local development fallback. "
            "Set BACKEND_URL if the backend is deployed separately."
        )

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

    st.subheader("Uploaded Documents")
    if not st.session_state.uploaded_files:
        st.info("No documents uploaded")
    else:
        for file in st.session_state.uploaded_files:
            st.write(f"📄 {file}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# MAIN INTERFACE
# =========================
st.title("📚 Enterprise Hybrid RAG")
st.caption("Production-grade Retrieval Augmented Generation System")
st.markdown("---")

# 1. Upload Section
st.subheader("📤 Upload Documents")
uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file and st.button("Upload PDF"):
    with st.spinner("Uploading document..."):
        if API_PREFIX is None:
            st.error(
                "Backend is not configured. Set BACKEND_URL to the deployed backend host before uploading files."
            )
        else:
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }
                response = requests.post(f"{API_PREFIX}/api/v1/upload", files=files, timeout=600)

                if response.status_code in [200, 201]:
                    st.success(f"Successfully uploaded {uploaded_file.name}")
                    if uploaded_file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(uploaded_file.name)
                else:
                    st.error(f"Upload Failed: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# 2. Query Section
st.subheader("💬 Ask Questions")
query = st.text_input("Enter your question")

if st.button("Ask") and query:
    with st.spinner("Retrieving answer..."):
        if API_PREFIX is None:
            st.error(
                "Backend is not configured. Set BACKEND_URL to the deployed backend host before asking questions."
            )
        else:
            payload = {"query": query, "top_k": 5, "stream": False}
            try:
                response = requests.post(
                    f"{API_PREFIX}/api/v1/ask", json=payload, timeout=300
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.messages.append({"question": query, "response": data})
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# 3. Chat History
for item in reversed(st.session_state.messages):
    st.markdown("---")
    st.markdown(f"### 🙋 Question\n{item['question']}")
    data = item["response"]

    st.markdown("### 🤖 Answer")
    st.write(data.get("answer", "No answer returned"))

    # Metrics
    st.markdown("### 📊 Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks Used", data.get("context_chunks_used", 0))
    with col2:
        latency = round(data.get("total_latency_ms", 0) / 1000, 2)
        st.metric("Latency (s)", latency)
    with col3:
        st.metric("Model", data.get("model", "N/A"))

    # Sources
    st.markdown("### 📚 Sources")
    citations = data.get("citations", [])
    if not citations:
        st.info("No citations returned")
    else:
        for index, citation in enumerate(citations, start=1):
            with st.expander(f"Source {index}"):
                st.write(f"📄 File: {citation.get('file_name', 'Unknown')}")
                st.write(f"📄 Page: {citation.get('page_number', 'N/A')}")
                st.write(f"📑 Title: {citation.get('title', 'N/A')}")
                st.write(f"⭐ Score: {round(citation.get('score', 0), 5)}")

st.markdown("---")
st.caption("Enterprise Hybrid RAG • FastAPI • Ollama • Streamlit")