from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Backend running successfully"}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # Get PORT from env (Render will set it)
    uvicorn.run("run:app", host="0.0.0.0", port=port)
