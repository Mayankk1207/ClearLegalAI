import os
import shutil
from pathlib import Path

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    HTTPException
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# -----------------------
# Directories
# -----------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


def reset_runtime_data():
    for directory in (UPLOAD_DIR, CHROMA_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


reset_runtime_data()

from rag.pipeline import chat
from rag.loader import process_pdf
from rag.vectorstore import clear_collection


# -----------------------
# Application
# -----------------------

app = FastAPI(
    title="ClearLegal AI",
    description="AI-powered legal information assistant",
    version="1.0.0",
)


# -----------------------
# Static Files & Templates
# -----------------------

if not STATIC_DIR.is_dir():
    raise RuntimeError(f"Static directory not found: {STATIC_DIR}")

if not TEMPLATES_DIR.is_dir():
    raise RuntimeError(f"Templates directory not found: {TEMPLATES_DIR}")

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="/static/favicon.svg")


# -----------------------
# Models
# -----------------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# -----------------------
# Pages
# -----------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/chat", response_class=HTMLResponse)
async def chatbot(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="chatbot.html",
    )


@app.get("/terms", response_class=HTMLResponse)
async def terms(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="terms.html",
    )


# -----------------------
# Health Check
# -----------------------

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": "ClearLegal AI",
    }


@app.post("/reset")
async def reset():
    for filepath in UPLOAD_DIR.iterdir():
        if filepath.is_file():
            filepath.unlink()
    clear_collection()

    return {"message": "Runtime data reset successfully."}


# -----------------------
# Upload PDF
# -----------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # Prevent directory traversal through uploaded filenames
    filename = Path(file.filename).name

    filepath = UPLOAD_DIR / filename

    try:

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        # Build the vector database
        process_pdf(str(filepath))

        return {
            "message": "PDF uploaded successfully."
        }

    except Exception as e:

        print(f"PDF upload error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the uploaded PDF."
        )

    finally:
        await file.close()


# -----------------------
# Chat API
# -----------------------

@app.post(
    "/api/chat",
    response_model=ChatResponse
)
async def chat_endpoint(req: ChatRequest):

    if not req.message.strip():

        return ChatResponse(
            reply="Please enter a legal question."
        )

    try:

        answer = chat(req.message)

        if not answer:

            answer = (
                "Sorry, I couldn't generate a response."
            )

    except Exception as e:

        print(f"Chat error: {e}")

        answer = (
            "Sorry, I'm unable to generate a response "
            "at the moment."
        )

    return ChatResponse(
        reply=answer
    )
