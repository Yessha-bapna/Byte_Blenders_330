# test_retriever.py

from app.services.retriever import get_top_chunks

query = "Does this policy cover air ambulance services?"
top_chunks = get_top_chunks(query, k=3)

print("\nTop Chunks:")
for i, chunk in enumerate(top_chunks, 1):
    print(f"\n--- Chunk {i} ---\n{chunk['text'][:300]}...")  # Print first 300 chars
