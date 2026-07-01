import os
import re
import json
import streamlit as st
import requests

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="Enterprise Hybrid RAG", page_icon="📚", layout="wide")

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://enterprise-hybrid-rag-1.onrender.com"
).strip().rstrip("/")
API_PREFIX = BACKEND_URL
backend_status = "configured"


def http_get_with_retry(url: str, retries: int = 3, delay: float = 1.0, timeout: int = 5):
    """Simple GET with retries for transient 5xx/connection errors."""
    import time
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if 500 <= resp.status_code < 600 and attempt < retries:
                time.sleep(delay * attempt)
                continue
            return resp
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay * attempt)
                continue
            raise


def http_post_with_retry(url: str, json=None, files=None, retries: int = 3, delay: float = 2.0, timeout: int = 60):
    """POST with retries on exceptions or transient 5xx responses."""
    import time
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=json, files=files, timeout=timeout)
            if resp.status_code == 429:
                return resp
            if 500 <= resp.status_code < 600 and attempt < retries:
                time.sleep(delay * attempt)
                continue
            return resp
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay * attempt)
                continue
            raise


def extract_backend_error_text(text: str) -> str:
    """Try to extract a clean error message from HTML or JSON backend responses."""
    text = text.strip()
    if not text:
        return "No response body returned from the backend."

    # Attempt JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return json.dumps(parsed, indent=2)
        return str(parsed)
    except json.JSONDecodeError:
        pass

    # Strip HTML tags if present
    if text.startswith("<!DOCTYPE html>") or text.startswith("<html"):
        clean = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.S)
        clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.S)
        clean = re.sub(r"<[^>]+>", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean[:4000] or "Backend returned an HTML error page."

    return text[:4000]

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
        health = http_get_with_retry(f"{API_PREFIX}/api/v1/health", retries=2, timeout=5)
        if health.status_code == 200:
            st.success("Backend Online")
        else:
            st.error(f"Backend Error: {health.status_code}")
            st.code(extract_backend_error_text(health.text))
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
            response = http_post_with_retry(f"{API_PREFIX}/api/v1/upload", files=files, timeout=600)

            if response.status_code in [200, 201]:
                st.success(f"Successfully uploaded {uploaded_file.name}")
                if uploaded_file.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(uploaded_file.name)
            else:
                st.error(f"Upload Failed (Status {response.status_code})")
                st.code(extract_backend_error_text(response.text))
        except Exception as e:
            st.error(f"Connection Error: {e}")

# 2. Query Section
st.subheader("💬 Ask Questions")
query = st.text_input("Enter your question")

if st.button("Ask") and query:
    with st.spinner("Retrieving answer..."):
        payload = {"query": query, "top_k": 5, "stream": False}
        try:
            response = http_post_with_retry(f"{API_PREFIX}/api/v1/ask", json=payload, retries=3, timeout=300)
            if response.status_code == 200:
                data = response.json()
                st.session_state.messages.append({"question": query, "response": data})
            else:
                st.error(f"Server Error ({response.status_code})")
                st.code(extract_backend_error_text(response.text))
        except Exception as e:
            st.error(f"Connection Error: {e}")

# 3. Chat History
for item in reversed(st.session_state.messages):
    st.markdown("---")
    st.markdown(f"### 🙋 Question\n{item['question']}")
    data = item["response"]
    st.markdown("### 🤖 Answer")
    st.write(data.get("answer", "No answer returned"))
