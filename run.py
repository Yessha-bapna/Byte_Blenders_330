from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

# ------------- Models and Classes -------------------

class QueryRequest(BaseModel):
    question: str
    chunks: list[str]

# ------------- Load models -------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_model = pipeline("text2text-generation", model="declare-lab/flan-alpaca-base", device=-1)

# ------------- Helper functions -------------------

def retrieve_top_chunks(question, chunks, k=3):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    c_emb = embedder.encode(chunks, convert_to_numpy=True)

    dim = c_emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(c_emb)

    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

def generate_answer(question, top_chunks):
    context_text = " ".join(top_chunks)
    context_text = context_text[:1600]  # limit ~400 tokens
    prompt = f"Answer the question based on the context:\n\nContext: {context_text}\n\nQuestion: {question}\nAnswer:"
    output = qa_model(prompt, max_new_tokens=256)
    return output[0]["generated_text"]

# ------------- Routes -------------------

@app.get("/")
def home():
    return {"message": "Backend running successfully"}

@app.post("/ask")
async def ask_question(req: QueryRequest):
    question = req.question
    chunks = req.chunks

    top_chunks = retrieve_top_chunks(question, chunks)
    answer = generate_answer(question, top_chunks)

    return {
        "answer": answer,
        "matched_clauses": top_chunks
    }

# ------------- Main Runner -------------------

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("run:app", host="0.0.0.0", port=port)
