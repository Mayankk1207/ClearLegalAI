import re
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .embedding import model
from .vectorstore import collection


SEMANTIC_CANDIDATES = 20
BM25_CANDIDATES = 20
RERANKED_DOCUMENTS = 4
RRF_K = 60

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _reciprocal_rank_fusion(
    semantic_documents: list[str],
    lexical_documents: list[str],
) -> list[str]:
    scores: dict[str, float] = {}

    for rank, document in enumerate(semantic_documents):
        scores[document] = scores.get(document, 0) + 1 / (RRF_K + rank + 1)

    for rank, document in enumerate(lexical_documents):
        scores[document] = scores.get(document, 0) + 1 / (RRF_K + rank + 1)

    return [
        document
        for document, _ in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def _bm25_documents(question: str) -> list[str]:
    stored = collection.get(include=["documents"])
    documents = [
        document
        for document in (stored.get("documents") or [])
        if isinstance(document, str) and document.strip()
    ]

    if not documents:
        return []

    corpus = [_tokenize(document) for document in documents]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(question))

    ranked_indexes = sorted(
        range(len(documents)),
        key=lambda index: scores[index],
        reverse=True,
    )

    return [
        documents[index]
        for index in ranked_indexes[:BM25_CANDIDATES]
    ]


def _semantic_documents(question: str) -> list[str]:
    vector = model.encode(question).tolist()
    result: dict[str, Any] = collection.query(
        query_embeddings=[vector],
        n_results=SEMANTIC_CANDIDATES,
    )
    documents = result.get("documents") or []

    if not documents:
        return []

    return [
        document
        for document in documents[0]
        if isinstance(document, str) and document.strip()
    ]


def retrieve(question: str) -> dict[str, list[list[str]]]:

    semantic_documents = _semantic_documents(question)
    lexical_documents = _bm25_documents(question)
    candidates = _reciprocal_rank_fusion(
        semantic_documents,
        lexical_documents,
    )

    if not candidates:
        return {"documents": [[]]}

    scores = reranker.predict([(question, document) for document in candidates])
    ranked_documents = [
        document
        for _, document in sorted(
            zip(scores, candidates),
            key=lambda item: item[0],
            reverse=True,
        )
    ]

    return {
        "documents": [ranked_documents[:RERANKED_DOCUMENTS]]
    }
