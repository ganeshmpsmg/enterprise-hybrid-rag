import streamlit as st
import requests

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="Enterprise Hybrid RAG", page_icon="📚", layout="wide")

# Ensure this is the correct URL for your live backend
BACKEND_URL = "https://enterprise-hybrid-rag.onrender.com"

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# =========================
# FUNCTIONS
# =========================
def get_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# =========================
# UI LAYOUT
# =========================
with st.sidebar:
    st.title("📚 Enterprise Hybrid RAG")
    if get_backend_status():
        st.success("Backend Online")
    else:
        st.error("Backend Offline / 502 Error")
        st.warning("Try refreshing in a minute. The free tier might be 'waking up'.")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("📚 Enterprise Hybrid RAG")

# 1. Upload Section
uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
if uploaded_file and st.button("Upload PDF"):
    with st.spinner("Processing document... (This may take a moment)"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            # Increased timeout for file processing
            response = requests.post(f"{BACKEND_URL}/api/v1/upload", files=files, timeout=300)
            
            if response.status_code == 200:
                st.success(f"Uploaded {uploaded_file.name}")
            else:
                st.error(f"Backend returned status {response.status_code}: {response.text}")
        except requests.exceptions.Timeout:
            st.error("Upload timed out. The server took too long to process the PDF.")
        except Exception as e:
            st.error(f"Connection failed: {e}")

# 2. Query Section
query = st.text_input("Ask a question about your documents:")
if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Increased timeout to 120s to allow backend to finish heavy inference
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/ask", 
                    json={"query": query, "top_k": 5}, 
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.messages.insert(0, {"question": query, "response": data})
                else:
                    st.error(f"Server Error ({response.status_code}). Check if the backend logs indicate a crash.")
            except requests.exceptions.Timeout:
                st.error("The model took too long to respond. Try a shorter query.")
            except Exception as e:
                st.error(f"Error: {e}")

# 3. Chat History
for item in st.session_state.messages:
    with st.container():
        st.markdown(f"**Q:** {item['question']}")
        st.write(item['response'].get("answer", "No content returned."))
        st.divider()
