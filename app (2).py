import firebase_admin
from firebase_admin import credentials, auth, db

import streamlit as st
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
from PyPDF2 import PdfReader
from docx import Document
from pathlib import Path
import re
import requests

# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(page_title="Persona Studio", layout="wide")

# ------------------------
# Firebase Setup
# ------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_config.json")  # your Firebase admin JSON
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://your-database-name.firebaseio.com/'  # Replace with your Firebase Realtime DB URL
    })

# ------------------------
# Constants
# ------------------------
DATA_DIR = Path("users")
DATA_DIR.mkdir(exist_ok=True)

FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")  # Web API key from Firebase project settings

# ------------------------
# OpenAI Setup (Backend-only)
# ------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # set as environment variable for security

# ------------------------
# Firebase Helper Functions
# ------------------------
def firebase_signup(email, password, name):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload).json()
    if 'error' in r:
        return None, r['error']['message']
    uid = r['localId']
    db.reference(f'/users/{uid}').set({"name": name, "email": email})
    return uid, None

def firebase_login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload).json()
    if 'error' in r:
        return None, r['error']['message']
    return r['idToken'], None

def verify_id_token(id_token):
    try:
        decoded = auth.verify_id_token(id_token)
        return decoded['uid']
    except:
        return None

# ------------------------
# Utility Functions
# ------------------------
def clean_text(t):
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'http\S+', '', t)
    return t.strip()

def chunk_text(text, chunk_size=500):
    text = text.strip()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def ensure_user_folder(user):
    user_dir = DATA_DIR / user
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def text_from_pdf(file_path):
    reader = PdfReader(str(file_path))
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def text_from_docx(file_path):
    doc = Document(str(file_path))
    return "\n".join([p.text for p in doc.paragraphs])

def save_persona(user, chunks, embeddings, index):
    user_dir = ensure_user_folder(user)
    with open(user_dir / "persona_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    faiss.write_index(index, str(user_dir / "persona_index.faiss"))

def load_persona(user):
    user_dir = DATA_DIR / user
    chunks_path = user_dir / "persona_chunks.json"
    index_path = user_dir / "persona_index.faiss"
    if chunks_path.exists() and index_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        index = faiss.read_index(str(index_path))
        return chunks, index
    return None, None

def build_persona_from_text(user, raw_text, model):
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, 500)
    if not chunks:
        return False, "No textual content found."
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    save_persona(user, chunks, embeddings, index)
    return True, f"Persona built with {len(chunks)} text chunks."

def retrieve_context(user, query, model, top_k=3):
    chunks, index = load_persona(user)
    if chunks is None:
        return []
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), top_k)
    return [chunks[i] for i in I[0] if 0 <= i < len(chunks)]

# ------------------------
# Load Embedding Model
# ------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embed_model()

# ------------------------
# Streamlit UI
# ------------------------
st.sidebar.title("Persona Studio")
menu = st.sidebar.radio("Menu", ["Login", "Sign Up"])

# ------------------------
# SIGNUP
# ------------------------
if menu == "Sign Up":
    st.header("Create a new account")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type="password")
    new_name = st.text_input("Full Name")
    if st.button("Sign Up"):
        if not new_email or not new_password or not new_name:
            st.warning("Fill all fields")
        else:
            uid, err = firebase_signup(new_email, new_password, new_name)
            if err:
                st.error(err)
            else:
                st.success("Account created! You can now log in.")
                st.stop()

# ------------------------
# LOGIN
# ------------------------
elif menu == "Login":
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        id_token, err = firebase_login(email, password)
        if err:
            st.error(err)
        else:
            uid = verify_id_token(id_token)
            if uid:
                st.session_state["uid"] = uid
                st.success(f"Logged in as {email}")
            else:
                st.error("Login failed")

# ------------------------
# Main App After Login
# ------------------------
if "uid" in st.session_state:
    uid = st.session_state["uid"]
    tabs = st.tabs(["Home", "Upload & Build", "Chat", "Manage"])

    with tabs[0]:
        st.header("Welcome to Persona Studio")
        st.markdown("Create a private AI version of yourself and chat with it!")

    with tabs[1]:
        st.header("Upload & Build Persona")
        file = st.file_uploader("Upload (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
        if file:
            user_dir = ensure_user_folder(uid)
            path = user_dir / file.name
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".txt"):
                text = open(path, "r", encoding="utf-8").read()
            elif file.name.endswith(".pdf"):
                text = text_from_pdf(path)
            else:
                text = text_from_docx(path)

            ok, msg = build_persona_from_text(uid, text, model)
            st.success(msg if ok else "Failed to build persona")

    with tabs[2]:
        st.header("Chat with your Persona")
        persona, idx = load_persona(uid)
        if persona is None:
            st.warning("No persona found. Build one first.")
        else:
            user_input = st.text_input("You:", "")
            if st.button("Send"):
                context = "\n".join(retrieve_context(uid, user_input, model))
                system = f"You are the AI version of this user.\nContext:\n{context}"
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_input}
                    ]
                )
                st.write("**AI Twin:**", resp.choices[0].message["content"])

    with tabs[3]:
        st.header("Manage Persona Files")
        user_dir = ensure_user_folder(uid)
        files = list(user_dir.glob("*"))
        for f in files:
            st.write(f.name, f.stat().st_size, "bytes")
        if st.button("Delete All"):
            for f in files:
                f.unlink()
            st.success("Deleted all persona files.")

    if st.sidebar.button("Logout"):
        st.session_state.pop("uid")
        st.success("Logged out")
        st.experimental_rerun()

