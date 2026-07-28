"""
MindSense AI - RAG Index Builder
===================================
Ingestion pipeline that:
  1. Scans the knowledge/ directory for PDF, TXT, and MD files
  2. Extracts text using PyMuPDF (PDF) or plain reading (TXT/MD)
  3. Splits text into overlapping chunks using LangChain's
     RecursiveCharacterTextSplitter
  4. Generates embeddings using SentenceTransformer
  5. Stores vectors in FAISS and saves metadata to JSON

Run this script once before starting the Flask server:

    python rag/build_index.py
    python rag/build_index.py --dry-run   # Test without saving
    python rag/build_index.py --reset     # Clear and rebuild index

Usage (Python API)::

    from rag.build_index import KnowledgeIndexBuilder
    builder = KnowledgeIndexBuilder()
    builder.build()
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # PyMuPDF

from config import settings
from model.faiss.index_manager import index_manager
from rag.embeddings import encode_texts
from utils.logger import get_logger
from utils.preprocessing import clean_text, normalize_whitespace
from utils.helpers import timer, hash_text

logger = get_logger(__name__)

# Try langchain splitter; fall back to our own implementation
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    logger.warning("langchain_text_splitters not found. Using built-in chunker.")


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        full_text = "\n\n".join(pages)
        logger.debug(f"Extracted {len(full_text)} chars from {pdf_path.name} ({len(pages)} pages)")
        return full_text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""


def extract_text_from_file(file_path: Path) -> str:
    """
    Extract text from a supported file (.pdf, .txt, .md).

    Args:
        file_path: Path to the file.

    Returns:
        Extracted text string. Empty string on failure.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)

    elif suffix in (".txt", ".md"):
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return ""
    else:
        logger.warning(f"Unsupported file type: {suffix} for {file_path.name}")
        return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, source_name: str) -> List[str]:
    """
    Split text into overlapping chunks using RecursiveCharacterTextSplitter.

    Args:
        text: Full document text to chunk.
        source_name: Source identifier for logging.

    Returns:
        List of text chunk strings.
    """
    chunk_size = settings.rag.CHUNK_SIZE
    chunk_overlap = settings.rag.CHUNK_OVERLAP
    min_length = settings.rag.MIN_CHUNK_LENGTH

    if HAS_LANGCHAIN:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )
        raw_chunks = splitter.split_text(text)
    else:
        # Built-in chunker fallback
        from utils.preprocessing import chunk_text_by_sentences
        raw_chunks = chunk_text_by_sentences(text, chunk_size, chunk_overlap)

    # Filter out chunks that are too short or empty
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) >= min_length]
    logger.debug(f"{source_name}: {len(raw_chunks)} raw chunks → {len(chunks)} after filtering")
    return chunks


# ---------------------------------------------------------------------------
# Main Index Builder
# ---------------------------------------------------------------------------

class KnowledgeIndexBuilder:
    """
    Builds the FAISS knowledge index from the structured knowledge base.

    Scans ``knowledge/`` directory tree and processes all supported files
    into embeddings stored in the FAISS index.

    Attributes:
        knowledge_dir (Path): Root of the knowledge base directory.
        dry_run (bool): If True, process but do not save the index.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.knowledge_dir: Path = settings.rag.KNOWLEDGE_DIR
        self.dry_run: bool = dry_run
        self._processed_hashes: set = set()  # Deduplication by content hash

    def discover_files(self) -> List[Tuple[Path, str]]:
        """
        Recursively discover all supported files in the knowledge directory.

        Returns:
            List of ``(file_path, category)`` tuples.
            Category is inferred from the immediate parent subdirectory name.
        """
        found: List[Tuple[Path, str]] = []
        supported = {ext.lower() for ext in settings.rag.SUPPORTED_EXTENSIONS}

        if not self.knowledge_dir.exists():
            logger.error(f"Knowledge directory not found: {self.knowledge_dir}")
            return []

        for category_dir in sorted(self.knowledge_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            category = category_dir.name

            for file_path in sorted(category_dir.rglob("*")):
                if file_path.suffix.lower() in supported and file_path.is_file():
                    found.append((file_path, category))

        logger.info(f"Discovered {len(found)} files across {self.knowledge_dir} knowledge base")
        return found

    def process_file(
        self,
        file_path: Path,
        category: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract, clean, and chunk a single knowledge file.

        Args:
            file_path: Path to the file.
            category: Knowledge category (CBT, DBT, etc.)

        Returns:
            Tuple of ``(chunks, metadata_entries)`` where metadata_entries
            contains one dict per chunk.
        """
        logger.info(f"Processing: [{category}] {file_path.name}")

        raw_text = extract_text_from_file(file_path)
        if not raw_text.strip():
            logger.warning(f"Empty content in {file_path.name} — skipping.")
            return [], []

        # Clean the extracted text
        cleaned = clean_text(
            raw_text,
            remove_url=False,   # Keep URLs in knowledge base
            remove_email=False,
            remove_phone=False,
            expand_contractions_flag=False,  # Don't modify clinical text
        )

        # Deduplicate by content hash (skip if already indexed)
        content_hash = hash_text(cleaned[:1000])
        if content_hash in self._processed_hashes:
            logger.debug(f"Duplicate content detected for {file_path.name} — skipping.")
            return [], []
        self._processed_hashes.add(content_hash)

        # Split into chunks
        chunks = chunk_text(cleaned, file_path.name)

        # Build metadata for each chunk
        metadata_entries = [
            {
                "text": chunk,
                "source": file_path.name,
                "source_path": str(file_path),
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_hash": content_hash,
            }
            for i, chunk in enumerate(chunks)
        ]

        return chunks, metadata_entries

    @timer
    def build(self, reset: bool = False) -> Dict[str, Any]:
        """
        Execute the full index building pipeline.

        Args:
            reset: If True, reset the existing index before building.

        Returns:
            Summary statistics dict with keys: total_files, total_chunks,
            total_vectors, categories, dry_run.
        """
        logger.info("=" * 60)
        logger.info("MindSense AI — Knowledge Index Build Pipeline")
        logger.info("=" * 60)

        if reset:
            index_manager.reset()
            logger.info("Existing index reset.")

        files = self.discover_files()
        if not files:
            logger.error(
                "No knowledge files found. Please add PDF/TXT/MD files "
                "to the knowledge/ subdirectories."
            )
            return {"error": "No knowledge files found"}

        all_chunks: List[str] = []
        all_metadata: List[Dict[str, Any]] = []
        stats_by_category: Dict[str, int] = {}
        failed_files = 0

        for file_path, category in files:
            try:
                chunks, metadata = self.process_file(file_path, category)
                all_chunks.extend(chunks)
                all_metadata.extend(metadata)
                stats_by_category[category] = stats_by_category.get(category, 0) + len(chunks)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                failed_files += 1

        if not all_chunks:
            logger.error("No chunks generated from knowledge base.")
            return {"error": "No chunks generated"}

        logger.info(f"Total chunks to embed: {len(all_chunks)}")

        # Generate embeddings in batches
        logger.info("Generating embeddings (this may take a few minutes)...")
        vectors = encode_texts(all_chunks, show_progress=True)

        logger.info(f"Generated embeddings: shape={vectors.shape}")

        if not self.dry_run:
            # Add to FAISS index
            index_manager.add_vectors(vectors, all_metadata)

            # Save index and metadata
            saved = index_manager.save()
            if saved:
                logger.info(f"Index saved to: {settings.faiss.INDEX_DIR}")
            else:
                logger.error("Failed to save index!")
        else:
            logger.info("[DRY RUN] Skipping save. Index not persisted.")

        summary = {
            "total_files": len(files),
            "failed_files": failed_files,
            "total_chunks": len(all_chunks),
            "total_vectors": len(vectors),
            "categories": stats_by_category,
            "dry_run": self.dry_run,
        }

        logger.info("=" * 60)
        logger.info("Index Build Complete:")
        for key, val in summary.items():
            logger.info(f"  {key}: {val}")
        logger.info("=" * 60)

        return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MindSense AI — Build the FAISS knowledge index from the knowledge base"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files and generate embeddings but do NOT save the index.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the existing index before rebuilding.",
    )
    args = parser.parse_args()

    builder = KnowledgeIndexBuilder(dry_run=args.dry_run)
    result = builder.build(reset=args.reset)

    if "error" in result:
        sys.exit(1)
