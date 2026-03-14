from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger
import threading

class MemoryEntry(BaseModel):
    """Represents a single memory entry."""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=lambda: __import__('time').time())

class MemoryStrategy(ABC):
    """Abstract base class for memory strategies."""
    
    @abstractmethod
    def add(self, entry: MemoryEntry, thread_id: Optional[str] = None):
        """Add an entry to memory."""
        pass

    @abstractmethod
    def retrieve(self, query: str, thread_id: Optional[str] = None, k: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant entries from memory."""
        pass

    @abstractmethod
    def clear(self, thread_id: Optional[str] = None):
        """Clear memory."""
        pass

class ShortTermMemory(MemoryStrategy):
    """Thread-based short-term memory (in-memory)."""
    
    def __init__(self):
        self._storage: Dict[str, List[MemoryEntry]] = {}
        self._lock = threading.Lock()

    def add(self, entry: MemoryEntry, thread_id: Optional[str] = None):
        if not thread_id:
            logger.warning("ShortTermMemory: No thread_id provided. Using 'default'.")
            thread_id = "default"
        
        with self._lock:
            if thread_id not in self._storage:
                self._storage[thread_id] = []
            self._storage[thread_id].append(entry)
            logger.debug(f"Added memory entry to thread {thread_id}")

    def retrieve(self, query: str, thread_id: Optional[str] = None, k: int = 5) -> List[MemoryEntry]:
        if not thread_id:
            thread_id = "default"
        
        with self._lock:
            entries = self._storage.get(thread_id, [])
            # For short-term, we just return the last k entries (chronological)
            return entries[-k:]

    def clear(self, thread_id: Optional[str] = None):
        with self._lock:
            if thread_id:
                self._storage.pop(thread_id, None)
            else:
                self._storage.clear()

class LongTermMemory(MemoryStrategy):
    """Vector-based long-term memory."""
    
    def __init__(self, collection_name: str = "agent_memory"):
        # Placeholder for real vector store initialization
        # In a real scenario, this would initialize ChromaDB or Pinecone
        self.collection_name = collection_name
        logger.info(f"Initialized LongTermMemory with collection: {collection_name}")

    def add(self, entry: MemoryEntry, thread_id: Optional[str] = None):
        # Implementation for vector store insertion
        logger.info(f"Persisting entry to long-term memory: {entry.content[:50]}...")
        # (ChromaDB/Pinecone code here)
        pass

    def retrieve(self, query: str, thread_id: Optional[str] = None, k: int = 5) -> List[MemoryEntry]:
        # Implementation for vector search
        logger.info(f"Searching long-term memory for: {query}")
        return []

class MemoryManager:
    """Singleton Memory Manager that orchestrates short and long term memory."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MemoryManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self._initialized = True
        logger.info("MemoryManager initialized successfully.")

    def add_memory(self, content: str, metadata: Dict[str, Any] = None, thread_id: str = None, long_term: bool = False):
        entry = MemoryEntry(content=content, metadata=metadata or {})
        self.short_term.add(entry, thread_id)
        if long_term:
            self.long_term.add(entry, thread_id)

    def get_context(self, query: str, thread_id: str = None, short_k: int = 10, long_k: int = 3) -> Dict[str, List[MemoryEntry]]:
        return {
            "short_term": self.short_term.retrieve(query, thread_id, k=short_k),
            "long_term": self.long_term.retrieve(query, thread_id, k=long_k)
        }
