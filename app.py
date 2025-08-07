import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
from docx import Document


# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="HackRx Policy QA", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
        .stApp { background-color: #0e0e0e; color: white; }
        div.stButton > button {
            background-color: #0d6efd; color: white; border: none; border-radius: 6px;
            padding: 0.7em 1.5em; font-weight: 600; font-size: 16px; width: 100%;
        }
        div.stButton > button:hover {
            background-color: white; color: #007bff; border: 1px solid #007bff;
        }
        .stTextInput input {
            background-color: #1a1a1a; color: white; border: 1px solid #007bff;
        }
        .stFileUploader label { color: white; }
        pre, code { color: #007bff; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Intelligent Policy Document Q&A")
st.write("Upload an insurance PDF and ask questions — get clause-level accurate answers.")

# ------------------ CACHE MODELS ------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_qa_model():
    return pipeline("text2text-generation", model="declare-lab/flan-alpaca-base", device=-1)  # CPU

embedder = load_embedder()
qa_model = load_qa_model()

# ------------------ FUNCTIONS ------------------

def load_and_chunk_file(file):
    chunks = []
    name = file.name.lower()

    if name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text = page.get_text("text")
            parts = text.split("\n\n")
            chunks.extend([p.strip() for p in parts if p.strip()])

    elif name.endswith(".docx"):
        document = Document(file)
        full_text = "\n".join([para.text for para in document.paragraphs])
        chunks = [p.strip() for p in full_text.split("\n\n") if p.strip()]

    elif name.endswith(".txt"):
        text = file.read().decode("utf-8")
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

    else:
        st.error("Unsupported file type. Please upload PDF, DOCX, or TXT.")
    
    return chunks


def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks

def retrieve_top_chunks(query, index, chunks, k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

def generate_answer(question, context_chunks, max_context_tokens=400):
    # Join chunks but limit total characters to avoid token overflow
    context_text = " ".join(context_chunks)
    context_text = context_text[:max_context_tokens * 4]  # ~4 chars per token

    prompt = f"Answer the question based on the context:\n\nContext: {context_text}\n\nQuestion: {question}\nAnswer:"
    response = qa_model(prompt, max_new_tokens=256)
    return response[0]['generated_text']

# ------------------ UI LAYOUT ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Upload a Policy Document")
    uploaded_file = st.file_uploader("Upload a policy document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

    if uploaded_file:
        with st.spinner("Processing document..."):
            chunks = load_and_chunk_file(uploaded_file)
            index, chunk_texts = build_faiss_index(chunks)
            st.session_state['index'] = index
            st.session_state['chunks'] = chunk_texts
        st.success(f"✅ Document processed with {len(chunk_texts)} chunks.")

with col2:
    st.subheader("2️⃣ Ask a Question")
    question = st.text_input("Type your question here")

    if st.button("Ask"):
        if 'index' not in st.session_state:
            st.warning("Please upload a document first.")
        elif question.strip():
            with st.spinner("Finding the best answer..."):
                # Retrieve only the top 1 best clause for display
                top_chunks = retrieve_top_chunks(question, st.session_state['index'], st.session_state['chunks'], k=1)
                answer = generate_answer(question, top_chunks)

                st.markdown(f"### ✅ Answer: {answer}")
                
                # Show only 1 best matched clause in preview
                st.markdown("**Matched Clause Preview:**")
                preview = top_chunks[0][:300] + ("..." if len(top_chunks[0]) > 300 else "")
                st.code(preview)
                with st.expander("📜 Show Full Clause"):
                    st.write(top_chunks[0])

                # JSON still contains all clauses from the retrieval (k=3)
                json_data = {
                    "question": question,
                    "answer": answer,
                    "matched_clauses": top_chunks
                }
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(json_data, indent=4),
                    file_name="response.json",
                    mime="application/json"
                )
        else:
            st.error("Please type a question.")
