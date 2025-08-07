from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF

app = FastAPI()

# Allow frontend (Streamlit) to access this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Streamlit domain for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Models ----------------

class QueryRequest(BaseModel):
    question: str
    chunks: list[str]

# ---------------- Load Models ----------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_model = pipeline("text2text-generation", model="declare-lab/flan-alpaca-base", device=-1)

# ---------------- Helper Functions ----------------

def retrieve_top_chunks(question, chunks, k=3):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    c_emb = embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(c_emb.shape[1])
    index.add(c_emb)
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

def generate_answer(question, top_chunks):
    context_text = " ".join(top_chunks)
    context_text = context_text[:1600]  # ~400 tokens
    prompt = f"Answer the question based on the context:\n\nContext: {context_text}\n\nQuestion: {question}\nAnswer:"
    output = qa_model(prompt, max_new_tokens=256)
    return output[0]["generated_text"]

def extract_chunks_from_pdf(file):
    doc = fitz.open(stream=file, filetype="pdf")
    chunks = []
    for page in doc:
        text = page.get_text("text")
        parts = text.split("\n\n")
        for part in parts:
            if part.strip():
                chunks.append(part.strip())
    return chunks

# ---------------- Routes ----------------

@app.get("/")
def home():
    return {"message": "Backend running successfully"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    chunks = extract_chunks_from_pdf(content)
    return {"chunks": chunks}

@app.post("/ask_uploaded")
async def ask_uploaded(req: QueryRequest):
    top_chunks = retrieve_top_chunks(req.question, req.chunks)
    answer = generate_answer(req.question, top_chunks)
    return {
        "answer": answer,
        "matched_clauses": top_chunks
    }

# (Optional) Still keeping original ask endpoint if used
@app.post("/ask")
async def ask_question(req: QueryRequest):
    top_chunks = retrieve_top_chunks(req.question, req.chunks)
    answer = generate_answer(req.question, top_chunks)
    return {
        "answer": answer,
        "matched_clauses": top_chunks
    }

# ---------------- Server Entry ----------------

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("run:app", host="0.0.0.0", port=port)
