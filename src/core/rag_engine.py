from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np
from rank_bm25 import BM25Okapi

class SearchResult:
    """Represents a single search result."""
    def __init__(self, content: str, score: float, metadata: Dict[str, Any] = None):
        self.content = content
        self.score = score
        self.metadata = metadata or {}

class BaseSearchStrategy(ABC):
    """Base class for search strategies."""
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        pass

class VectorSearch(BaseSearchStrategy):
    """Semantic vector search strategy."""
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        logger.debug(f"Executing vector search for: {query}")
        # Implementation for ChromaDB/Pinecone search
        return []

class BM25Search(BaseSearchStrategy):
    """Keyword-based BM25 search strategy."""
    def __init__(self, documents: List[str]):
        self.tokenized_corpus = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.documents = documents

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        logger.debug(f"Executing BM25 search for: {query}")
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:k]
        
        results = []
        for i in top_n:
            if scores[i] > 0:
                results.append(SearchResult(content=self.documents[i], score=scores[i]))
        return results

class Reranker:
    """Reranks search results using a cross-encoder or other logic."""
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        logger.info(f"Reranking {len(results)} results for query: {query}")
        # In a production-grade system, this would use a cross-encoder model
        # For now, we'll just return as-is
        return sorted(results, key=lambda x: x.score, reverse=True)

class RAGEngine:
    """Production-grade RAG engine with hybrid search and reranking."""
    
    def __init__(self, vector_store: Optional[VectorSearch] = None, bm25_store: Optional[BM25Search] = None):
        self.vector_store = vector_store or VectorSearch()
        self.bm25_store = bm25_store
        self.reranker = Reranker()

    def query(self, query_text: str, k: int = 5, hybrid_weight: float = 0.5) -> List[SearchResult]:
        """
        Executes a hybrid search and reranks the results.
        
        Args:
            query_text: The query string.
            k: Number of results to return.
            hybrid_weight: Weight given to vector search (0 to 1).
        """
        logger.info(f"RAGEngine query: {query_text}")
        
        vector_results = self.vector_store.search(query_text, k=k*2)
        bm25_results = self.bm25_store.search(query_text, k=k*2) if self.bm25_store else []
        
        # Combine results (Reciprocal Rank Fusion or simple weighted sum)
        combined_results = self._combine_results(vector_results, bm25_results, hybrid_weight)
        
        # Rerank
        final_results = self.reranker.rerank(query_text, combined_results)
        
        return final_results[:k]

    def _combine_results(self, vector_res: List[SearchResult], bm25_res: List[SearchResult], weight: float) -> List[SearchResult]:
        # Simple combination logic for the prototype
        # In a real system, you'd normalize scores or use RRF
        seen_content = {}
        
        for res in vector_res:
            seen_content[res.content] = res.score * weight
            
        for res in bm25_res:
            if res.content in seen_content:
                seen_content[res.content] += res.score * (1 - weight)
            else:
                seen_content[res.content] = res.score * (1 - weight)
                
        return [SearchResult(content=k, score=v) for k, v in seen_content.items()]
