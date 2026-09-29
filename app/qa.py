import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app import config
from app.vectorstore import search

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You answer questions using only the context below. "
        "The answer may be worded differently from the question, so read the context carefully. "
        "Only if the context has no relevant information, reply exactly: insufficient information. "
        "Do not use outside knowledge and do not guess.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])


def build_context(results):
    parts = []
    for doc, _ in results:
        header = f"[{doc.metadata['file_name']}, page {doc.metadata['page_number']}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question):
    results = search(question, config.TOP_K)
    for doc, score in results:
        logger.info("retrieved %s (distance %.3f)", doc.metadata["chunk_id"], score)

    if not results:
        return {"answer": "insufficient information", "citations": []}

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
        timeout=30,
        max_retries=2,
    )
    messages = PROMPT.format_messages(context=build_context(results), question=question)
    logger.info("calling LLM with %d chunks", len(results))
    answer = llm.invoke(messages).content.strip()

    if "insufficient information" in answer.lower():
        return {"answer": answer, "citations": []}

    citations = [
        {
            "file_name": doc.metadata["file_name"],
            "page_number": doc.metadata["page_number"],
            "chunk_id": doc.metadata["chunk_id"],
        }
        for doc, _ in results
    ]
    return {"answer": answer, "citations": citations}