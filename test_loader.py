# test_loader.py

from app.services.document_loader import load_and_chunk_pdf

chunks = load_and_chunk_pdf("data/EDLHLGA23009V012223.pdf")
print(f"Extracted {len(chunks)} chunks")
print(chunks[0])  # print first chunk
