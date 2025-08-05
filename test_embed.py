# test_embed.py

from app.services.document_loader import load_and_chunk_pdf
from app.services.embedder import embed_chunks, save_faiss_index

chunks = load_and_chunk_pdf("data/EDLHLGA23009V012223.pdf")
embeddings = embed_chunks(chunks)
save_faiss_index(embeddings, chunks)
print("FAISS index created and saved!")
