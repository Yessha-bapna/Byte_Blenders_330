# app/services/qa_engine.py
from sentence_transformers import SentenceTransformer
import numpy as np
from app.services.retriever import get_top_chunks
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
model_id = "declare-lab/flan-alpaca-base"
qa_pipeline = pipeline("text2text-generation", model=model_id, tokenizer=model_id)

def format_prompt(context, question):
    return f"""
Answer the following insurance question based only on the context:

### Context:
{context}

### Question:
{question}

### Format:
Answer: <Yes/No/Partially>
Reason: <short explanation>
Matched Clause: <quote exact sentence>

Now answer:
"""

def generate_answer(question, k=3):
    top_chunks = get_top_chunks(question, k=k)
    context = "\n".join([f"{chunk['document']} - Chunk {i+1}: {chunk['text']}" for i, chunk in enumerate(top_chunks)])

    prompt = format_prompt(context, question)
    result = qa_pipeline(prompt, max_length=512, do_sample=False)[0]['generated_text']

    return {
        "question": question,
        "answer": result,
        "source_document": top_chunks[0]["document"],
        "matched_chunk": top_chunks[0]["text"]
    }

def ask_uploaded_question(question, session, k=3):
    model_id = "declare-lab/flan-alpaca-base"
    qa_pipeline = pipeline("text2text-generation", model=model_id, tokenizer=model_id)

    index = session["index"]
    chunks = session["chunks"]

    # Embed the query
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    q_embedding = embed_model.encode([question])
    D, I = index.search(np.array(q_embedding), k)

    top_chunks = [chunks[i] for i in I[0]]

    context = "\n".join([f"{chunk['text']}" for chunk in top_chunks])

    prompt = f"""
Answer the following insurance question based only on the context:

### Context:
{context}

### Question:
{question}

### Format:
Answer: <Yes/No/Partially>
Reason: <short explanation>
Matched Clause: <quote exact sentence>

Now answer:
"""

    result = qa_pipeline(prompt, max_length=512, do_sample=False)[0]['generated_text']

    return {
        "question": question,
        "answer": result,
        "matched_clause": top_chunks[0]["text"],
        "document": session["doc_name"]
    }

