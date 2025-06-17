import streamlit as st
import openai
import pickle
import faiss
import numpy as np

# -------------- AUTH ----------------
USERS = st.secrets["users"]


def login():
    st.title("🔐 LGA Handbook Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state['authenticated'] = True
            st.experimental_rerun()
        else:
            st.error("Invalid credentials.")

if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
    login()
    st.stop()

# -------------- HEADER ----------------
st.set_page_config(page_title="LGA Handbook AI Search", page_icon="📘")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("lga_logo.png", width=80)  # ✅ place logo in the same folder as app.py
with col2:
    st.title("LGA Handbook AI Search")
    st.write("Ask me anything! If there is something in the LGA Handbook relevant to your question, I will let you know.")

# -------------- OPENAI ----------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -------------- INDEX LOADING ----------------
INDEX_FILE = "vector.index"
CHUNKS_FILE = "chunks.pkl"

@st.cache_resource
def load_index_and_chunks():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

index, chunks = load_index_and_chunks()

# -------------- EMBEDDING ----------------
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# -------------- SEARCH ----------------
def search_chunks(query, index, chunks, top_k=5):
    query_vector = get_embedding(query)
    D, I = index.search(np.array([query_vector]).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

# -------------- GPT ----------------
def ask_gpt(question, context_chunks):
    context = ""
    for c in context_chunks:
        context += f"{c['text']}\n\n(Source: {c['source']}, Page {c.get('page', '?')})\n\n"

    prompt = f"""
You are an AI trained to help legal professionals find relevant content in the LGA Handbook.
Use only direct quotes from the context and cite them clearly.

Context:
{context}

Question:
{question}
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# -------------- UI ----------------
query = st.text_input("Enter your question here.")
if st.button("Submit") and query:
    with st.spinner("Searching LGA Handbook..."):
        matches = search_chunks(query, index, chunks)
        answer = ask_gpt(query, matches)
    st.markdown("### 📘 Answer:")
    st.write(answer)
