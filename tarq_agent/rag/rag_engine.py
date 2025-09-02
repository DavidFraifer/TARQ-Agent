"""
RAG Engine for TARQ Agent
Handles document ingestion and context retrieval for enhanced LLM responses.
"""

import os
import numpy as np
import pdfplumber
from typing import List, Dict, Optional

# Suppress TensorFlow warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings

# Try FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Don't import sentence_transformers at module level - use lazy loading
SENTENCE_TRANSFORMERS_AVAILABLE = True
try:
    import sentence_transformers
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..utils.console import console

# Configuration
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 40


class VectorStore:
    """Vector storage and search using FAISS."""
    
    def __init__(self, dim=EMBED_DIM):
        self.dim = dim
        self.metadata = []
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for RAG functionality. Install with: pip install faiss-cpu")
        
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray, metas: list):
        """Add vectors and metadata to the store."""
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metas)

    def search(self, qvec: np.ndarray, k=5):
        """Search for similar vectors."""
        faiss.normalize_L2(qvec)
        scores, idxs = self.index.search(qvec, k)
        
        results = []
        for s, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(self.metadata):
                md = self.metadata[i]
                results.append({
                    "score": float(s),
                    "text": md["text"],
                    "doc_id": md.get("doc_id")
                })
        
        return results


class RAGEngine:
    """RAG Engine for document ingestion and context retrieval."""
    
    def __init__(self):
        self.embedder = None
        self.store = None
        self.enabled = False
        self._model_loaded = False
        
        # Check if dependencies are available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            console.warning("RAG", "sentence-transformers not available. RAG disabled.")
            return
            
        if not FAISS_AVAILABLE:
            console.warning("RAG", "FAISS not available. RAG disabled. Install with: pip install faiss-cpu")
            return
            
        # Initialize vector store immediately (lightweight)
        try:
            self.store = VectorStore()
            self.enabled = True
            console.success("RAG", "RAG engine initialized successfully")
        except Exception as e:
            console.error("RAG", f"Failed to initialize RAG engine: {e}")

    def _ensure_model_loaded(self):
        """Lazy load the embedder model only when needed."""
        if self._model_loaded or not self.enabled:
            return
            
        try:
            console.info("RAG", "Loading embedder model...")
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
            self._model_loaded = True
        except Exception as e:
            console.error("RAG", f"Failed to load embedder model: {e}")
            self.enabled = False

    def is_enabled(self) -> bool:
        """Check if RAG is enabled and ready to use."""
        return self.enabled

    def _choose_chunk_params(self, text: str):
        """Choose chunk size & overlap dynamically based on text length."""
        num_words = len(text.split())

        if num_words < 2000:       # ~1–3 pages
            return 180, 30
        elif num_words < 6000:     # ~4–10 pages
            return 220, 40
        else:                      # longer docs
            return 260, 50

    def _chunk_text(self, text: str, chunk_size: int, overlap: int):
        """Split text into overlapping chunks."""
        tokens = text.split()
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunks.append(" ".join(tokens[start:end]))
            if end == len(tokens):
                break
            start = end - overlap

        return chunks

    def _extract_text_from_pdf(self, path: str) -> str:
        """Extract full text from a PDF file."""
        try:
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages)
        except Exception as e:
            console.error("RAG", f"Failed to extract text from PDF {path}: {e}")
            return ""

    def ingest_text(self, text: str, doc_id: str = None):
        """Ingest text content into the RAG system."""
        if not self.enabled:
            return
            
        # Ensure model is loaded before processing
        self._ensure_model_loaded()
        if not self._model_loaded:
            return
            
        try:
            # Decide chunk params dynamically
            chunk_size, overlap = self._choose_chunk_params(text)
            console.info("RAG", f"Using chunk_size={chunk_size}, overlap={overlap}")

            # Split text and embed
            chunks = self._chunk_text(text, chunk_size, overlap)
            embeddings = self.embedder.encode(
                chunks, convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")

            # Attach metadata
            metas = [{
                "id": f"{doc_id or 'doc'}_{i}",
                "text": chunks[i],
                "doc_id": doc_id
            } for i in range(len(chunks))]

            self.store.add(embeddings, metas)
            console.success("RAG", f"Added {len(chunks)} chunks from {doc_id or 'document'}")
            
        except Exception as e:
            console.error("RAG", f"Failed to ingest text: {e}")

    def ingest_file(self, path: str, doc_id: str = None):
        """Ingest a file into the RAG system."""
        if not self.enabled:
            return
            
        if not os.path.exists(path):
            console.error("RAG", f"File not found: {path}")
            return

        try:
            if path.lower().endswith('.pdf'):
                text = self._extract_text_from_pdf(path)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()

            if text.strip():
                self.ingest_text(text, doc_id=doc_id or os.path.basename(path))
            else:
                console.warning("RAG", f"No text extracted from {path}")
                
        except Exception as e:
            console.error("RAG", f"Failed to ingest file {path}: {e}")

    def query(self, question: str, k: int = 3) -> List[Dict]:
        """Query the RAG system for relevant context."""
        if not self.enabled:
            return []
            
        # Ensure model is loaded before querying
        self._ensure_model_loaded()
        if not self._model_loaded:
            return []
            
        try:
            q_embed = self.embedder.encode([question], convert_to_numpy=True).astype("float32")
            results = self.store.search(q_embed, k=k)
            
            # Filter results with reasonable similarity scores
            filtered_results = [r for r in results if r.get("score", 0) > 0.3]
            
            if filtered_results:
                console.info("RAG", f"Found {len(filtered_results)} relevant context chunks")
            
            return filtered_results
            
        except Exception as e:
            console.error("RAG", f"Failed to query RAG: {e}")
            return []

    def get_context(self, question: str, max_length: int = 1000) -> str:
        """Get formatted context string for LLM prompting."""
        if not self.enabled:
            return ""
            
        results = self.query(question)
        if not results:
            return ""
            
        context_parts = []
        current_length = 0
        
        for result in results:
            text = result["text"]
            if current_length + len(text) > max_length:
                # Truncate to fit
                remaining = max_length - current_length
                if remaining > 100:  # Only add if meaningful chunk remains
                    text = text[:remaining] + "..."
                    context_parts.append(text)
                break
            
            context_parts.append(text)
            current_length += len(text)
        
        if context_parts:
            return "\n\n".join(context_parts)
        return ""
