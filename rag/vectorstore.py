from pathlib import Path

import chromadb


CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    "legal_docs"
)

def add_chunks(chunks, embeddings):

    collection.add(

        ids=[
            str(i)
            for i in range(len(chunks))
        ],

        documents=[
            c["text"]
            for c in chunks
        ],

        embeddings=embeddings,

        metadatas=[
            {
                "page": c["page"]
            }
            for c in chunks
        ]
    )