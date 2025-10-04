from typing import List, Dict, Any
import logging
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from config import DB_DIR, DB_COLLECTION, OPENAI_API_KEY, OPENAI_API_BASE, EMBEDDING_MODEL
from .embeddings import get_embeddings

_client = chromadb.PersistentClient(path=DB_DIR)
_collection = _client.get_or_create_collection(name=DB_COLLECTION)
_log = logging.getLogger(__name__)

embedding_client = OpenAIEmbeddings(
	model=EMBEDDING_MODEL,
	openai_api_key=OPENAI_API_KEY,
	openai_api_base=OPENAI_API_BASE,
)

def add_chunks_to_vectorstore(chunks: List[Dict[str, Any]]) -> None:
	if not chunks:
		return
	texts = [ch["content"] for ch in chunks]
	ids = [str(ch["chunk_id"]) for ch in chunks]
	metadatas = [{
		"doc_title": ch.get("doc_title"),
		"section": ch.get("section"),
		"chunk_id": ch.get("chunk_id"),
	} for ch in chunks]

	embs = get_embeddings(texts)
	if not embs or len(embs) != len(texts):
		print("Failed to compute embeddings for some or all chunks; aborting save.")
		return

	_collection.add(documents=texts, embeddings=embs, metadatas=metadatas, ids=ids)
	print(f"Saved {len(chunks)} chunks to ChromaDB collection '{DB_COLLECTION}'.")

def get_langchain_chroma_retriever(persist_directory=DB_DIR, collection_name=DB_COLLECTION):
	"""
	Returns a LangChain Chroma retriever for the specified collection.
	"""
	try:
		vectorstore = Chroma(
			collection_name=collection_name,
			persist_directory=persist_directory,
			embedding_function=embedding_client
		)
	except Exception as e:
		_log.error(f"Error while loading Chroma vectorstore: {e}")
		return None

	return vectorstore.as_retriever(search_type="similarity",
                search_kwargs={
                    "k": 4
                })

def get_langchain_chroma_vectorstore(persist_directory=DB_DIR, collection_name=DB_COLLECTION):
	"""
	Returns a LangChain Chroma vectorstore for the specified collection.
	"""
	try:
		vectorstore = Chroma(
			collection_name=collection_name,
			persist_directory=persist_directory,
			embedding_function=embedding_client
		)
	except Exception as e:
		_log.error(f"Error while loading Chroma vectorstore: {e}")
		return None

	return vectorstore
