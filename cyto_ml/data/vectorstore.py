import chromadb
from chromadb.db.base import UniqueConstraintError
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)

client = chromadb.PersistentClient(path="./vectors")


def vector_store(name: Optional[str] = "test_collection"):
    """
    Return a vector store specified by name, default test_collection
    """
    try:
        collection = client.create_collection(
            name=name, metadata={"hnsw:space": "cosine"}  # default similarity
        )
    except UniqueConstraintError as err:
        collection = client.get_collection(name)
        logging.info(err)

    return collection
