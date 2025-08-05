# app/routes.py

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict
import uuid

from app.services.document_loader import load_and_chunk_pdf
from app.services.embedder import embed_chunks
from app.services.qa_engine import ask_uploaded_question

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

router = APIRouter()

# Temporary in-memory store
upload_sessions: Dict[str, dict] = {}

# --- Upload endpoint ---
@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_bytes = await file.read()
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    # Process
    chunks = load_and_chunk_pdf(temp_path)
    embeddings = embed_chunks(chunks)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    session_id = str(uuid.uuid4())[:8]
    upload_sessions[session_id] = {
        "index": index,
        "chunks": chunks,
        "doc_name": file.filename
    }

    return {"message": "PDF uploaded successfully", "session_id": session_id}


# --- Ask from uploaded ---
class UploadedAskRequest(BaseModel):
    session_id: str
    question: str

@router.post("/ask_uploaded")
def ask_uploaded(payload: UploadedAskRequest):
    session = upload_sessions.get(payload.session_id)
    if not session:
        return {"error": "Invalid session ID"}

    answer = ask_uploaded_question(payload.question, session)
    return answer
