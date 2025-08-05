# app/services/retriever.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the same embedding model used during indexing
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the saved FAISS index
def load_faiss_index(index_path="vector.index"):
    index = faiss.read_index(index_path)
    return index

# Load metadata
def load_metadata(metadata_path="vector_metadata.txt"):
    metadata = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|||", 1)
            if len(parts) == 2:
                chunk_id = parts[0]
                text = parts[1]
                doc_name = chunk_id.split("_chunk_")[0]  # Get original filename
                metadata.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "document": doc_name
                })
    return metadata

# Retrieve top-k similar chunks
def get_top_chunks(query, k=3):
    index = load_faiss_index()
    metadata = load_metadata()

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)

    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results
