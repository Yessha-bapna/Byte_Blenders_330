# build_index.py

import os
from app.services.document_loader import load_and_chunk_pdf
from app.services.embedder import embed_chunks, save_faiss_index

DATA_DIR = "data"
all_chunks = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pdf"):
        path = os.path.join(DATA_DIR, filename)
        chunks = load_and_chunk_pdf(path)
        all_chunks.extend(chunks)

print(f"Total chunks from all PDFs: {len(all_chunks)}")
embeddings = embed_chunks(all_chunks)
save_faiss_index(embeddings, all_chunks)
