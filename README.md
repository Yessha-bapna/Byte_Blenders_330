# 📄 HackRx Policy Q&A System

Welcome to our HackRx 6.0 project -- an intelligent document-based
question answering system designed specifically for insurance policy
documents. Our solution allows users to upload any insurance policy in
PDF format and ask natural language questions to get highly accurate,
clause-level answers.

## 🚀 Problem Statement

Policy documents are lengthy, jargon-heavy, and hard to understand for
the average user. Customers struggle to find specific information,
leading to poor decision-making or delayed support. We aim to solve this
using LLM-powered intelligent Q&A on insurance policies.

## 💡 Solution

We built a Retrieval-Augmented Generation (RAG)-based system using a
local open-source LLM (Flan-Alpaca) that semantically retrieves the
relevant clause and generates the answer. It includes:\
- PDF ingestion and chunking\
- Embedding using MiniLM\
- FAISS vector search\
- LLM-based answer generation\
- A full-stack app with FastAPI backend and Streamlit frontend

## 📂 Folder Structure

\`\`\`\
hackrx_rag_project/\
│\
├── app/ \# Backend app logic\
│ ├── main.py \# FastAPI entrypoint\
│ ├── routes.py \# API routes\
│ ├── services/ \# Core logic\
│ │ ├── document_loader.py\
│ │ ├── qa_engine.py\
├── build_index.py \# Preload index builder\
├── app.py \# Streamlit frontend\
└── README.md \# This file\
\`\`\`

## ⚙️ How to Run

\*\*1. Install Requirements:\*\*\
\`\`\`bash\
pip install -r requirements.txt\
\`\`\`\
\*\*2. Start Backend:\*\*\
\`\`\`bash\
python -m uvicorn app.main:app \--reload\
\`\`\`\
\*\*3. Start Frontend:\*\*\
\`\`\`bash\
streamlit run app.py\
\`\`\`

## 📌 Features

\- Upload multiple insurance policy PDFs\
- Ask clause-specific questions in natural language\
- Works offline with open-source LLM\
- Clause reference and document name shown with every answer\
- Download Q&A in JSON format

## 🧠 Tech Stack

\- \*\*Frontend:\*\* Streamlit\
- \*\*Backend:\*\* FastAPI, Uvicorn\
- \*\*LLM:\*\* Flan-Alpaca (HuggingFace)\
- \*\*Embedding:\*\* MiniLM (SentenceTransformers)\
- \*\*Vector DB:\*\* FAISS\
- \*\*Language:\*\* Python 3.10

## 🏆 Why This Project Stands Out

\- Local LLM (no OpenAI dependency)\
- Full offline RAG pipeline\
- Real-world use case with high applicability\
- Intuitive UI and API\
- Built in just 2 days for HackRx 6.0

## 👥 Team

\- Team: \[Byte_Blenders\]\
- Hackathon: Bajaj Finserv HackRx 6.0
