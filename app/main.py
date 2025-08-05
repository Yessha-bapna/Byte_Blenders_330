# app/main.py

from fastapi import FastAPI, Request
from app.routes import router

app = FastAPI(title="HackRx Retrieval QA API")

app.include_router(router)
