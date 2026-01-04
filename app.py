__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import os
import requests
import hashlib
from docx import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- 1. CONFIGURATION & SECURITY ---
st.set_page_config(page_title="AI Vault", layout="wide")

SAVE_DIR = "vault_storage"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Default PIN: 123456 (You can change this)
PIN_HASH = hashlib.sha256("123456".encode()).hexdigest()

def check_auth():
    if "auth" not in st.session_state:
        st.session_state.auth = False
    
    if not st.session_state.auth:
        st.title("🔐 Secure AI Vault")
        t1, t2 = st.tabs(["Fingerprint / Biometric", "Backup PIN"])
        
        with t1:
            st.info("Biometrics require HTTPS (Streamlit Cloud).")
            if st.button("Simulate Fingerprint Scan"):
                st.session_state.auth = True
                st.rerun()
        
        with t2:
            pin = st.text_input("Enter 6-Digit PIN", type="password", max_chars=6)
            if len(pin) == 6:
                if hashlib.sha256(pin.encode()).hexdigest() == PIN_HASH:
                    st.session_state.auth = True
                    st.rerun()
                else:
                    st.error("Wrong PIN")
        st.stop()

check_auth()

# --- 2. HELPER FUNCTIONS ---
def resolve_and_download(url):
    try:
        res = requests.get(url, allow_redirects=True, timeout=10)
        final_url = res.url
        if "docs.google.com/spreadsheets" in final_url:
            final_url = final_url.split("/edit")[0] + "/export?format=csv"
            res = requests.get(final_url)
        
        fname = final_url.split("/")[-1].split("?")[0] or "link_data.csv"
        path = os.path.join(SAVE_DIR, fname)
        with open(path, "wb") as f:
            f.write(res.content)
        return fname
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("📂 Data Management")
    if st.button("Logout"):
        st.session_state.auth = False
        st.rerun()
    
    st.divider()
    up = st.file_uploader("Upload File", type=["docx", "xlsx", "csv"])
    if up:
        with open(os.path.join(SAVE_DIR, up.name), "wb") as f:
            f.write(up.getbuffer())
        st.success("Uploaded!")

    link = st.text_input("🔗 Paste Link (Short/Google/Direct)")
    if st.button("Fetch Link"):
        name = resolve_and_download(link)
        if name: 
            st.success(f"Saved: {name}")
            st.rerun()

    st.divider()
    files = os.listdir(SAVE_DIR)
    selected = st.selectbox("Open File", ["None"] + files)

# --- 4. MAIN INTERFACE ---
if selected != "None":
    path = os.path.join(SAVE_DIR, selected)
    content = ""
    
    if selected.endswith(".docx"):
        content = "\n".join([p.text for p in Document(path).paragraphs])
        st.text_area("Word Content", content, height=200)
    elif selected.endswith((".xlsx", ".csv")):
        df = pd.read_excel(path) if selected.endswith(".xlsx") else pd.read_csv(path)
        content = df.to_string()
        st.dataframe(df)

    st.subheader("🤖 Ask AI")
    key = st.text_input("OpenAI API Key", type="password")
    if key and content:
        # Optimized Chunking
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(content)
        
        # Vector Search
        embeddings = OpenAIEmbeddings(openai_api_key=key)
        vector = Chroma.from_texts(chunks, embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=key, model="gpt-3.5-turbo"), 
            retriever=vector.as_retriever()
        )
        
        query = st.text_input("Search details or ask a question:")
        if query:
            with st.spinner("AI is analyzing..."):
                result = qa.invoke(query)
                st.info(result["result"])
