import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app import config

logger = logging.getLogger(__name__)

store = None


def get_store():
    global store
    if store is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )
        store = Chroma(
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory=config.CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return store


def add_chunks(chunks, file_name):
    db = get_store()

    db.delete(where={"file_name": file_name})

    ids = []
    for chunk in chunks:
        ids.append(chunk.metadata["chunk_id"])

    db.add_documents(chunks, ids=ids)
    logger.info("stored %d chunks for %s", len(chunks), file_name)


def search(question, k):
    db = get_store()
    return db.similarity_search_with_score(question, k=k)


def count_chunks():
    db = get_store()
    return get_store()._collection.count()