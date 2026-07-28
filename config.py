"""
MindSense AI - Centralized Configuration
=========================================
Provides a single source of truth for all configuration parameters.
Uses python-dotenv for environment variable management and dataclasses
for type-safe, documented configuration objects.

Author: MindSense AI Team
Version: 1.0.0
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project root directory (absolute path)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Application Configuration
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    """Flask application configuration."""

    # Flask settings
    SECRET_KEY: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "mindsense-secret-key-change-in-production"))
    DEBUG: bool = field(default_factory=lambda: os.getenv("FLASK_DEBUG", "False").lower() == "true")
    HOST: str = field(default_factory=lambda: os.getenv("FLASK_HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: int(os.getenv("FLASK_PORT", "5000")))
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16 MB max upload

    # Application metadata
    APP_NAME: str = "MindSense AI"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI-Powered Mental Health Assistant"


# ---------------------------------------------------------------------------
# Gemini LLM Configuration
# ---------------------------------------------------------------------------
@dataclass
class GeminiConfig:
    """Google Gemini API configuration."""

    API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    MODEL_NAME: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    TEMPERATURE: float = field(default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.7")))
    TOP_P: float = field(default_factory=lambda: float(os.getenv("GEMINI_TOP_P", "0.95")))
    TOP_K: int = field(default_factory=lambda: int(os.getenv("GEMINI_TOP_K", "40")))
    MAX_OUTPUT_TOKENS: int = field(default_factory=lambda: int(os.getenv("GEMINI_MAX_TOKENS", "2048")))
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.5  # seconds between retries
    STREAM: bool = True  # Enable streaming responses


# ---------------------------------------------------------------------------
# Embedding Model Configuration
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingConfig:
    """Sentence-Transformers embedding model configuration."""

    MODEL_NAME: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    MODEL_DIR: Path = BASE_DIR / "model" / "embedding_model"
    DEVICE: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    BATCH_SIZE: int = 32
    NORMALIZE_EMBEDDINGS: bool = True
    EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 output dimension


# ---------------------------------------------------------------------------
# Classifier Configuration
# ---------------------------------------------------------------------------
@dataclass
class ClassifierConfig:
    """Mental health classifier configuration."""

    MODEL_DIR: Path = BASE_DIR / "model" / "classifier"
    # Supported output labels (7 classes)
    LABELS: List[str] = field(
        default_factory=lambda: [
            "Normal",
            "Stress",
            "Depression",
            "Anxiety",
            "Suicidal",
            "Bipolar",
            "Personality Disorder",
        ]
    )
    CONFIDENCE_THRESHOLD: float = 0.3  # Minimum confidence to trust prediction
    # High-risk labels that trigger crisis protocol
    HIGH_RISK_LABELS: List[str] = field(
        default_factory=lambda: ["Suicidal", "Depression", "Bipolar"]
    )


# ---------------------------------------------------------------------------
# FAISS Vector Store Configuration
# ---------------------------------------------------------------------------
@dataclass
class FAISSConfig:
    """FAISS vector index configuration."""

    INDEX_DIR: Path = BASE_DIR / "model" / "faiss"
    INDEX_FILE: str = "knowledge_index.faiss"
    METADATA_FILE: str = "metadata.json"
    INDEX_TYPE: str = "Flat"  # "Flat" (exact) or "IVF" (approximate, faster for large corpora)
    NLIST: int = 100  # Number of Voronoi cells for IVF index
    NPROBE: int = 10  # Number of cells to search during query


# ---------------------------------------------------------------------------
# RAG (Retrieval-Augmented Generation) Configuration
# ---------------------------------------------------------------------------
@dataclass
class RAGConfig:
    """RAG pipeline configuration."""

    KNOWLEDGE_DIR: Path = BASE_DIR / "knowledge"
    # Chunking parameters
    CHUNK_SIZE: int = 500          # Characters per chunk
    CHUNK_OVERLAP: int = 100       # Overlap between consecutive chunks
    # Retrieval parameters
    TOP_K_RETRIEVE: int = 10       # Initial FAISS retrieval count
    TOP_K_RERANK: int = 5          # Final chunks after reranking
    # Reranker model
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_DEVICE: str = field(default_factory=lambda: os.getenv("RERANKER_DEVICE", "cpu"))
    # Knowledge base categories
    KNOWLEDGE_CATEGORIES: List[str] = field(
        default_factory=lambda: ["CBT", "DBT", "WHO", "APA", "Coping", "Crisis", "TherapistExamples"]
    )
    MIN_CHUNK_LENGTH: int = 50     # Minimum characters for a chunk to be indexed
    SUPPORTED_EXTENSIONS: List[str] = field(default_factory=lambda: [".pdf", ".txt", ".md"])


# ---------------------------------------------------------------------------
# Memory Configuration
# ---------------------------------------------------------------------------
@dataclass
class MemoryConfig:
    """Conversation and session memory configuration."""

    MEMORY_DIR: Path = BASE_DIR / "memory"
    USER_PROFILES_DIR: Path = BASE_DIR / "memory" / "user_profiles"
    MAX_CONVERSATION_TURNS: int = 20   # Max turns stored in-session
    MAX_SESSION_SUMMARY_LENGTH: int = 500  # Characters for session summary
    SUMMARY_TRIGGER_TURNS: int = 10    # Summarize after N turns


# ---------------------------------------------------------------------------
# Response Validation Configuration
# ---------------------------------------------------------------------------
@dataclass
class ValidationConfig:
    """Response quality validation thresholds."""

    # Scoring weights (must sum to 1.0)
    EMPATHY_WEIGHT: float = 0.25
    RELEVANCE_WEIGHT: float = 0.25
    SAFETY_WEIGHT: float = 0.25
    GROUNDEDNESS_WEIGHT: float = 0.15
    LENGTH_WEIGHT: float = 0.05
    REPETITION_WEIGHT: float = 0.05

    # Quality threshold (0.0 – 1.0); below this triggers regeneration
    QUALITY_THRESHOLD: float = 0.60
    MAX_REGENERATION_ATTEMPTS: int = 2

    # Response length bounds (characters)
    MIN_RESPONSE_LENGTH: int = 80
    MAX_RESPONSE_LENGTH: int = 2500

    # Unsafe / crisis keywords that trigger immediate override
    UNSAFE_KEYWORDS: List[str] = field(
        default_factory=lambda: [
            "kill yourself",
            "end your life",
            "you should die",
            "you are worthless",
            "no one cares",
            "just give up",
        ]
    )


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
@dataclass
class LoggingConfig:
    """Structured logging configuration."""

    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: str = "mindsense.log"
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    MAX_BYTES: int = 10 * 1024 * 1024   # 10 MB per log file
    BACKUP_COUNT: int = 5
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Risk Assessment Configuration
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    """Risk assessment configuration for the risk agent."""

    # Risk levels
    RISK_LEVELS: List[str] = field(default_factory=lambda: ["low", "moderate", "high", "critical"])

    # Keyword sets for risk scoring (case-insensitive)
    CRITICAL_KEYWORDS: List[str] = field(
        default_factory=lambda: [
            "suicide", "suicidal", "kill myself", "end my life", "want to die",
            "no reason to live", "better off dead", "overdose", "self-harm",
            "cutting myself", "hurting myself", "plan to die",
        ]
    )
    HIGH_KEYWORDS: List[str] = field(
        default_factory=lambda: [
            "hopeless", "worthless", "trapped", "can't go on", "no way out",
            "burden to everyone", "nobody cares", "give up on everything",
        ]
    )
    MODERATE_KEYWORDS: List[str] = field(
        default_factory=lambda: [
            "depressed", "anxious", "panic attack", "crying all the time",
            "can't sleep", "exhausted", "overwhelmed", "losing control",
        ]
    )

    # Risk score weights
    CRITICAL_SCORE: float = 1.0
    HIGH_SCORE: float = 0.7
    MODERATE_SCORE: float = 0.4
    LOW_SCORE: float = 0.1


# ---------------------------------------------------------------------------
# Master Settings object — single import point for the entire project
# ---------------------------------------------------------------------------
class Settings:
    """
    Master configuration container.
    Import this class everywhere in the project.

    Usage::

        from config import settings
        print(settings.gemini.MODEL_NAME)
        print(settings.rag.TOP_K_RERANK)
    """

    def __init__(self) -> None:
        self.app = AppConfig()
        self.gemini = GeminiConfig()
        self.embedding = EmbeddingConfig()
        self.classifier = ClassifierConfig()
        self.faiss = FAISSConfig()
        self.rag = RAGConfig()
        self.memory = MemoryConfig()
        self.validation = ValidationConfig()
        self.logging = LoggingConfig()
        self.risk = RiskConfig()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create all required directories if they don't already exist."""
        dirs = [
            self.logging.LOG_DIR,
            self.faiss.INDEX_DIR,
            self.memory.USER_PROFILES_DIR,
            self.embedding.MODEL_DIR,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """
        Validate critical configuration values.
        Raises ValueError if any required setting is missing.
        """
        if not self.gemini.API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add it to your .env file.\n"
                "Example: GEMINI_API_KEY=your_api_key_here"
            )
        logging.info("Configuration validated successfully.")


# ---------------------------------------------------------------------------
# Global singleton — import `settings` everywhere
# ---------------------------------------------------------------------------
settings = Settings()
