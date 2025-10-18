from typing import List, Dict, Any, Optional, Union
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from uuid import uuid4

from config import (
    DB_DIR,
    DB_COLLECTION,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    EMBEDDING_MODEL,
)

_log = logging.getLogger(__name__)

embedding_client = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Create a Qdrant client for local storage
client = QdrantClient(path=DB_DIR)
try:
	client.create_collection(
		collection_name=DB_COLLECTION,
		vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
		sparse_vectors_config={
			"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
		},
	)
except Exception as e:
	_log.warning(f"Collection '{DB_COLLECTION}' already exists: {e}")

qdrant = QdrantVectorStore(
    client=client,
    collection_name=DB_COLLECTION,
    embedding=embedding_client,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

def add_chunks_to_vectorstore(
    chunks: List[Dict[str, Any]],
    collection_name: Optional[str] = None,
) -> None:
    """
    Add text chunks with metadata to Qdrant vectorstore (using LangChain Qdrant integration).
    """
    if not chunks:
        _log.info("No chunks provided; skipping vectorstore save.")
        return

    texts = [ch["content"] for ch in chunks]
    ids = [str(uuid4()) for ch in chunks]
    metadatas = [
        {
            "doc_title": ch.get("doc_title"),
            "section": ch.get("section"),
            "chunk_id": ch.get("chunk_id"),
        }
        for ch in chunks
    ]

    try:
        qdrant.add_texts(texts, metadatas=metadatas, ids=ids)
        _log.info("Saved %d chunks to Qdrant collection '%s'.", len(chunks), collection_name or DB_COLLECTION)
    except Exception as e:
        _log.error("Error while saving to Qdrant: %s", e)


def get_qdrant_vectorstore(
    collection_name: Optional[str] = DB_COLLECTION,
    search_type: str = "dense",
):
    """
    Return a LangChain Qdrant vectorstore.
    """
    try:
        if search_type == "sparse":
        	return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                sparse_embedding=sparse_embeddings,
                retrieval_mode=RetrievalMode.SPARSE,
                sparse_vector_name="sparse"
            )
        elif search_type == "dense":
        	return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedding_client,
                retrieval_mode=RetrievalMode.DENSE,
                vector_name="dense",
            )
        elif search_type == "hybrid":
                return QdrantVectorStore(
				client=client,
				collection_name=collection_name,
				embedding=embedding_client,
				sparse_embedding=sparse_embeddings,
				retrieval_mode=RetrievalMode.HYBRID,
				vector_name="dense",
				sparse_vector_name="sparse",
			)
        else:
            _log.error("Invalid search_type '%s'; must be 'dense', 'sparse', or 'hybrid'.", search_type)
            return None
        
    except Exception as e:
        _log.error("Error while initializing Qdrant vectorstore: %s", e)
        return None

def get_qdrant_retriever(collection_name=DB_COLLECTION, search_type: str = "dense",):
	"""
	Returns a LangChain Qdrant retriever for the specified collection.
	"""
	try:
		vectorstore = get_qdrant_vectorstore(
			collection_name=collection_name,
			search_type=search_type,
		)
	except Exception as e:
		_log.error(f"Error while loading Qdrant vectorstore: {e}")
		return None

	if vectorstore is None:
		_log.error(f"Failed to initialize Qdrant vectorstore.")
		return None

	return vectorstore.as_retriever(search_type="similarity",
                search_kwargs={
                    "k": 6
                })
