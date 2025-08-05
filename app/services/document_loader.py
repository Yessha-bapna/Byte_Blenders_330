# app/services/document_loader.py
import os 
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    return chunks

def load_and_chunk_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    return [{
        "chunk_id": f"{os.path.basename(pdf_path)}_chunk_{i}",
        "text": chunk,
        "document": os.path.basename(pdf_path)
    } for i, chunk in enumerate(chunks)]
