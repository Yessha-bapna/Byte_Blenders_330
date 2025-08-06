# run.py
import os
import uvicorn

if __name__ == "__main__":
    # Render provides a $PORT environment variable
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 for local testing
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=bool(os.environ.get("RELOAD", False))  # Reload only if RELOAD env is set
    )
