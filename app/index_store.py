"""FAISS vector index storage, metadata mapping, and manifest serialization."""

import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class IndexManifest:
    build_timestamp: str
    model_name: str
    pretrained: str
    embedding_dimension: int
    vector_count: int
    open_clip_version: str
    torch_version: str
    catalog_source: str
    skipped_count: int
    failures: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IndexStore:
    """Manages building, saving, and loading FAISS index and metadata mapping."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ):
        self.index_path = index_path or settings.index_path
        self.metadata_path = metadata_path or settings.metadata_path
        self.manifest_path = manifest_path or settings.manifest_path

    def exists(self) -> bool:
        """Check if both index and metadata files exist."""
        return self.index_path.is_file() and self.metadata_path.is_file()

    def build_index(self, vectors: np.ndarray) -> faiss.IndexFlatIP:
        """Create a FAISS IndexFlatIP from normalized float32 vectors."""
        if len(vectors) == 0:
            raise ValueError("Cannot build index with zero vectors.")
        
        dim = vectors.shape[1]
        # Inner Product with normalized vectors corresponds to Cosine Similarity
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))
        return index

    def save(
        self,
        index: faiss.Index,
        metadata: List[Dict[str, Any]],
        manifest: IndexManifest,
    ) -> None:
        """Atomically save index, metadata mapping, and manifest."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary files first for atomic replacement
        temp_dir = self.index_path.parent

        # 1. Save FAISS index
        temp_index = temp_dir / f"tmp_{self.index_path.name}"
        faiss.write_index(index, str(temp_index))
        if self.index_path.exists():
            self.index_path.unlink()
        temp_index.rename(self.index_path)

        # 2. Save metadata JSON
        temp_meta = temp_dir / f"tmp_{self.metadata_path.name}"
        with open(temp_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        temp_meta.rename(self.metadata_path)

        # 3. Save manifest JSON
        temp_man = temp_dir / f"tmp_{self.manifest_path.name}"
        with open(temp_man, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        temp_man.rename(self.manifest_path)

        logger.info(
            f"Successfully saved index ({index.ntotal} vectors), metadata ({len(metadata)} items), "
            f"and manifest to {temp_dir}"
        )

    def load(self) -> Tuple[faiss.Index, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Load index, metadata, and manifest.
        
        Raises:
            FileNotFoundError: If index or metadata file is not found.
            RuntimeError: If files cannot be parsed.
        """
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"Vector index not found at '{self.index_path}'. "
                "Please run 'python scripts/build_index.py' to generate the catalog index."
            )
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Index metadata not found at '{self.metadata_path}'. "
                "Please run 'python scripts/build_index.py' to generate the catalog index."
            )

        try:
            index = faiss.read_index(str(self.index_path))
        except Exception as e:
            raise RuntimeError(f"Failed to read FAISS index from '{self.index_path}': {e}") from e

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata JSON from '{self.metadata_path}': {e}") from e

        manifest = None
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read index manifest: {e}")

        if index.ntotal != len(metadata):
            logger.warning(
                f"Index vector count ({index.ntotal}) differs from metadata item count ({len(metadata)})."
            )

        return index, metadata, manifest

    def search(
        self,
        index: faiss.Index,
        query_vector: np.ndarray,
        top_k: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the FAISS index with a single query vector.
        
        Args:
            index: FAISS index
            query_vector: shape (dimension,) or (1, dimension), float32
            top_k: number of nearest neighbors to retrieve

        Returns:
            Tuple of (scores: np.ndarray, indices: np.ndarray) of shape (k,)
        """
        if index.ntotal == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        
        # Ensure query is float32
        query_vector = query_vector.astype(np.float32)

        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vector, k)
        return scores[0], indices[0]

