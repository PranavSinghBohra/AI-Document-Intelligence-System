import logging
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app import config, qa, vectorstore  
from app.pdf_processor import process_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Document Intelligence System")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    result = {
        "status": "ok",
        "openai_key_configured": bool(config.OPENAI_API_KEY),
        "vector_store": "ok",
        "chunks_indexed": None,
    }
    try:
        result["chunks_indexed"] = vectorstore.count_chunks()
    except Exception:
        logger.exception("health check could not reach the vector store")
        result["vector_store"] = "error"
    if not result["openai_key_configured"] or result["vector_store"] == "error":
        result["status"] = "degraded"
    return result


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    original_name = file.filename
    if original_name is None or original_name == "":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_name = os.path.basename(original_name)
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    with open(file_path, "wb") as saved_file:
        shutil.copyfileobj(file.file, saved_file)
    logger.info("upload received: %s", file_name)

    if os.path.getsize(file_path) == 0:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="The file is empty")

    try:
        chunks = process_pdf(file_path, file_name)
    except ValueError as error:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("could not read %s", file_name)
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Could not read the PDF, it may be corrupt")

    try:
        vectorstore.add_chunks(chunks, file_name)
    except Exception:
        logger.exception("could not store chunks for %s", file_name)
        raise HTTPException(status_code=503, detail="Could not store the document, try again later")

    return {"file_name": file_name, "chunks_stored": len(chunks)}


@app.post("/ask")
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    logger.info("question received: %s", question)

    try:
        if vectorstore.count_chunks() == 0:
            return {"answer": "No documents uploaded yet. Upload a PDF first.", "citations": []}
        return qa.answer_question(question)
    except Exception:
        logger.exception("could not answer the question")
        raise HTTPException(status_code=503, detail="Could not answer right now, try again later")