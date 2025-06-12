import streamlit as st
import pickle
import faiss
import openai
import numpy as np
import os

# Set your API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load cached index and chunks
INDEX_FILE = "vector.index"
CHUNKS_FILE = "chunks.pkl"

@st.cache_resource
def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search_chunks(query, index, chunks, top_k=5):
    query_vector = get_embedding(query)
    D, I = index.search(np.array([query_vector]).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

def ask_gpt(question, context_chunks):
    context = ""
    for c in context_chunks:
        context += f"{c['text']}\n\n(Source: {c['source']}, Page {c.get('page', '?')})\n\n"

    prompt = f"""You are a legal assistant AI. You must answer questions using only direct quotes from the provided context. Cite each quote by document name and page number and also quote the relevant sections. If no quote can be found, respond: 'No source found in provided documents.'

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

# ----------------- UI ------------------

st.set_page_config(page_title="Legal PDF AI", layout="wide")
st.title("📚 Legal Document Assistant")
st.markdown("Ask questions based on your uploaded legal PDFs.")

query = st.text_input("Enter your legal question:")

if query:
    with st.spinner("Searching and generating response..."):
        index, chunks = load_index()
        results = search_chunks(query, index, chunks)
        answer = ask_gpt(query, results)

        st.markdown("### 🧠 Answer")
        st.write(answer)

        with st.expander("📄 Source Context"):
            for chunk in results:
                st.markdown(f"**{chunk['source']} - Page {chunk['page']}**")
                st.markdown(f"> {chunk['text']}")
