import os
import uvicorn

SERVICE_HOST = os.environ.get("SERVICE_HOST", "127.0.0.1")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host=SERVICE_HOST, port=8000)
