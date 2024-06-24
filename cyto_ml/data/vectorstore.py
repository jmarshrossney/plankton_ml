import chromadb
from chromadb.db.base import NotFoundError, UniqueConstraintError
from typing import Optional

client = chromadb.PersistentClient(path="./vectors")

def vector_store(name: Optional[str] = 'test_collection'):
    """
    Return a vector store specified by name, default test_collection
    """
    try:
        collection = client.create_collection(
            name=name, metadata={"hnsw:space": "cosine"}  # l2 is the default
        )
    except UniqueConstraintError as err:
        collection = client.get_collection(name)
    
    return collection
