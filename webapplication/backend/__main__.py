import uvicorn
from .config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run("webapplication.backend.main:app", host=HOST, port=PORT, reload=False)
