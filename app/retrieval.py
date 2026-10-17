"""Vector retrieval and candidate aggregation by canonical model ID."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.catalog import CatalogStore
from app.config import settings
from app.index_store import IndexStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedCandidate:
    model_id: str
    model_number: str
    product_title: str
    image_similarity: float
    best_image_id: str
    best_image_path: Optional[str] = None
    product_url: Optional[str] = None
    image_url: Optional[str] = None
    category: Optional[str] = None
    finish: Optional[str] = None
    handle_configuration: Optional[str] = None
    mounting_type: Optional[str] = None
    spout_type: Optional[str] = None
    specifications_json: Optional[str] = None
    inventory_status: Optional[str] = None
    source_name: Optional[str] = None
    image_hits_count: int = 1
    hit_scores: List[float] = field(default_factory=list)


class Retriever:
    """Performs vector search in the FAISS index and aggregates candidates by model ID."""

    def __init__(
        self,
        index_store: Optional[IndexStore] = None,
        catalog_store: Optional[CatalogStore] = None,
    ):
        self.index_store = index_store or IndexStore()
        self.catalog_store = catalog_store or CatalogStore()
        self.index = None
        self.metadata = None
        self.manifest = None
        self._loaded = False

    def load(self) -> None:
        """Load FAISS index and metadata."""
        if not self._loaded:
            self.index, self.metadata, self.manifest = self.index_store.load()
            self._loaded = True

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k_images: Optional[int] = None,
    ) -> List[RetrievedCandidate]:
        """Search vector index and aggregate results by canonical model_id.

        Args:
            query_vector: L2-normalized float32 vector.
            top_k_images: Number of raw image vectors to retrieve (default: 50).

        Returns:
            List of unique RetrievedCandidate models sorted by image_similarity descending.
        """
        self.load()

        if self.index is None or self.index.ntotal == 0:
            logger.warning("Retrieval attempted on empty or uninitialized index.")
            return []

        top_k = top_k_images or settings.retrieval_top_k
        scores, indices = self.index_store.search(self.index, query_vector, top_k=top_k)

        # Aggregate by model_id: keep highest score and record hits
        model_hits: Dict[str, Dict[str, Any]] = {}

        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            sim_score = float(score)
            item_meta = self.metadata[idx]
            model_id = str(item_meta.get("model_id"))

            if model_id not in model_hits:
                model_hits[model_id] = {
                    "model_id": model_id,
                    "model_number": item_meta.get("model_number", model_id),
                    "best_score": sim_score,
                    "best_image_id": item_meta.get("image_id", ""),
                    "best_image_path": item_meta.get("local_path"),
                    "product_url": item_meta.get("product_url"),
                    "product_title": item_meta.get("product_title", ""),
                    "scores": [sim_score],
                }
            else:
                record = model_hits[model_id]
                record["scores"].append(sim_score)
                if sim_score > record["best_score"]:
                    record["best_score"] = sim_score
                    record["best_image_id"] = item_meta.get("image_id", "")
                    record["best_image_path"] = item_meta.get("local_path")
                    if not record["product_url"]:
                        record["product_url"] = item_meta.get("product_url")

        # Sort aggregated models by highest image similarity descending
        sorted_records = sorted(
            model_hits.values(), key=lambda x: x["best_score"], reverse=True
        )

        candidates: List[RetrievedCandidate] = []
        for rec in sorted_records:
            m_id = rec["model_id"]
            db_product = self.catalog_store.get_product(m_id) or {}

            candidate = RetrievedCandidate(
                model_id=m_id,
                model_number=db_product.get("model_number") or rec["model_number"],
                product_title=db_product.get("product_title") or rec["product_title"],
                image_similarity=rec["best_score"],
                best_image_id=rec["best_image_id"],
                best_image_path=rec["best_image_path"] or db_product.get("local_path"),
                product_url=db_product.get("product_url") or rec["product_url"],
                image_url=db_product.get("image_url"),
                category=db_product.get("category"),
                finish=db_product.get("finish"),
                handle_configuration=db_product.get("handle_configuration"),
                mounting_type=db_product.get("mounting_type"),
                spout_type=db_product.get("spout_type"),
                specifications_json=db_product.get("specifications_json"),
                inventory_status=db_product.get("inventory_status"),
                source_name=db_product.get("source_name"),
                image_hits_count=len(rec["scores"]),
                hit_scores=rec["scores"],
            )
            candidates.append(candidate)

        return candidates

