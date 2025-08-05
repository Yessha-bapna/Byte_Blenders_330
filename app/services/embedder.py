# app/services/embedder.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and good

def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts)
    return embeddings

def save_faiss_index(embeddings, chunks, index_path="vector.index"):
    import os

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_path)

    # Safe write of metadata: each chunk in one line
    with open("vector_metadata.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            clean_text = chunk["text"].replace("\n", " ").replace("\r", " ").strip()
            line = f"{chunk['chunk_id']}|||{clean_text}"
            f.write(line + "\n")

