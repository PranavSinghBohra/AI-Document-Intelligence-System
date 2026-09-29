from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import config

def clean_text(text):
    text = text.replace("-\n", "")

    cleaned_paragraphs = []
    for paragraph in text.split("\n\n"):
        paragraph = " ".join(paragraph.split())
        if paragraph != "":
            cleaned_paragraphs.append(paragraph)

    return "\n\n".join(cleaned_paragraphs)


def process_pdf(file_path, file_name):
    try:
        pages = PyMuPDFLoader(file_path).load()
    except Exception:
        raise ValueError("Please upload a valid PDF.")

    pages_with_text = []
    for page in pages:
        page.page_content = clean_text(page.page_content)
        if page.page_content != "":
            pages_with_text.append(page)

    if len(pages_with_text) == 0:
        raise ValueError("No text found in the PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages_with_text)

    for i, chunk in enumerate(chunks):
        page_number = chunk.metadata["page"] + 1
        chunk.metadata = {
            "file_name": file_name,
            "page_number": page_number,
            "chunk_id": f"{file_name}_p{page_number}_c{i}",
        }

    return chunks