"""
RAG (Retrieval-Augmented Generation) module for TARQ Agent.
Provides document ingestion and context retrieval capabilities.
"""

# Suppress TensorFlow warnings before any RAG imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from .rag_engine import RAGEngine

__all__ = ['RAGEngine']
