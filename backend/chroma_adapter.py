
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import logging

logger = logging.getLogger(__name__)

embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

class ChromaManager:
    def __init__(self, path="chroma_db", collection_name="default_collection"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        self.collection_name = collection_name

    def clear_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Successfully cleared and re-initialized collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Could not clear collection '{self.collection_name}': {e}")

    def add_documents(self, docs: list[str], metadatas: list[dict], ids: list[str]):
        try:
            self.collection.add(documents=docs, metadatas=metadatas, ids=ids)
            logger.info(f"Added {len(docs)} documents to Chroma collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to add documents to chroma collection '{self.collection_name}': {e}")

    def query(self, query_text: str, n_results: int = 5):
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
            formatted = []
            if results and results.get("metadatas"):
                for meta, dist, doc in zip(results.get("metadatas", [[]])[0], results.get("distances", [[]])[0], results.get("documents", [[]])[0]):
                    meta = dict(meta or {})
                    meta['score'] = float(dist)
                    meta['content'] = doc
                    formatted.append(meta)
            logger.info(f"Found {len(formatted)} relevant results in collection '{self.collection_name}'.")
            return formatted
        except Exception as e:
            logger.error(f"Chroma query failed for collection '{self.collection_name}': {e}")
            return []

    def get_collection_count(self):
        return self.collection.count()
