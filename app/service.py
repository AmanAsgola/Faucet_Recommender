"""End-to-end orchestration service for faucet identification and recommendation."""

import logging
from dataclasses import asdict, dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image

from app.catalog import CatalogStore
from app.config import settings
from app.embeddings import ImageEmbedder
from app.image_utils import validate_and_load_image
from app.index_store import IndexStore
from app.rerank import HybridReranker, RerankedCandidate
from app.retrieval import Retriever

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    status: str  # "success", "no_match", "uncertain", "error"
    overall_confidence: str  # likely_match, needs_confirmation, no_reliable_match
    guidance_message: str
    candidates: List[Dict[str, Any]]
    disclaimer: str
    query_preview_available: bool = True
    error_message: Optional[str] = None


class FaucetIdentificationService:
    """Coordinates image validation, embedding generation, vector retrieval, and reranking."""

    def __init__(
        self,
        embedder: Optional[ImageEmbedder] = None,
        retriever: Optional[Retriever] = None,
        reranker: Optional[HybridReranker] = None,
        catalog_store: Optional[CatalogStore] = None,
        index_store: Optional[IndexStore] = None,
    ):
        self.catalog_store = catalog_store or CatalogStore()
        self.index_store = index_store or IndexStore()
        self.embedder = embedder  # Lazy loaded on first search to keep app start fast
        self.retriever = retriever or Retriever(
            index_store=self.index_store, catalog_store=self.catalog_store
        )
        self.reranker = reranker or HybridReranker()

    def _get_embedder(self) -> ImageEmbedder:
        if self.embedder is None:
            self.embedder = ImageEmbedder.get_instance()
        return self.embedder

    def check_system_ready(self) -> Tuple[bool, str]:
        """Check if catalog and index are built and ready for queries."""
        if not self.index_store.exists():
            return False, (
                "Catalog vector index is missing. Please run 'python scripts/build_index.py' "
                "to generate the FAISS index."
            )
        if self.catalog_store.count_products() == 0:
            return False, (
                "Catalog database is empty. Please run 'python scripts/validate_catalog.py' "
                "and 'python scripts/build_index.py'."
            )
        return True, "System ready."

    def identify(
        self,
        image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
        query_text: Optional[str] = None,
        query_attributes: Optional[Dict[str, str]] = None,
        use_hybrid_reranking: bool = True,
        top_k: int = 3,
    ) -> IdentificationResult:
        """Process an uploaded faucet image and return top matches with evidence and uncertainty.

        Args:
            image_input: Uploaded file stream, bytes, file path, or PIL Image.
            query_text: Optional user query string (e.g. "Single handle kitchen pull-down").
            query_attributes: Optional known attributes (e.g. {"category": "Kitchen", "finish": "Chrome"}).
            use_hybrid_reranking: Whether to use hybrid reranking (defaults to True).
            top_k: Number of candidates to return (defaults to 3).

        Returns:
            IdentificationResult with top matches, confidence, and guidance.
        """
        # 1. System readiness check
        ready, msg = self.check_system_ready()
        if not ready:
            return IdentificationResult(
                status="error",
                overall_confidence=settings.label_no_reliable_match,
                guidance_message=msg,
                candidates=[],
                disclaimer=settings.repair_part_disclaimer,
                query_preview_available=False,
                error_message=msg,
            )

        # 2. Image validation and preprocessing
        try:
            if isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            else:
                img = validate_and_load_image(image_input)
        except Exception as e:
            return IdentificationResult(
                status="error",
                overall_confidence=settings.label_no_reliable_match,
                guidance_message="Image validation failed. Please upload a valid JPEG, PNG, or WebP photo.",
                candidates=[],
                disclaimer=settings.repair_part_disclaimer,
                query_preview_available=False,
                error_message=str(e),
            )

        # 3. Vector embedding
        try:
            embedder = self._get_embedder()
            query_vec = embedder.embed_image(img)
            
            query_text_vec = None
            if use_hybrid_reranking and query_text and query_text.strip():
                query_text_vec = embedder.embed_text(query_text.strip())
        except Exception as e:
            logger.exception("Error generating embedding.")
            return IdentificationResult(
                status="error",
                overall_confidence=settings.label_no_reliable_match,
                guidance_message="An error occurred while generating visual features.",
                candidates=[],
                disclaimer=settings.repair_part_disclaimer,
                query_preview_available=True,
                error_message=str(e),
            )

        # 4. Vector retrieval
        try:
            raw_candidates = self.retriever.retrieve(query_vec)
        except Exception as e:
            logger.exception("Error during index retrieval.")
            return IdentificationResult(
                status="error",
                overall_confidence=settings.label_no_reliable_match,
                guidance_message="Failed to retrieve matching products from vector index.",
                candidates=[],
                disclaimer=settings.repair_part_disclaimer,
                query_preview_available=True,
                error_message=str(e),
            )

        if not raw_candidates:
            return IdentificationResult(
                status="no_match",
                overall_confidence=settings.label_no_reliable_match,
                guidance_message=(
                    "No matching faucet models found in the catalog. "
                    "Try uploading a clearer photo showing the faucet base, handles, or model label."
                ),
                candidates=[],
                disclaimer=settings.repair_part_disclaimer,
                query_preview_available=True,
            )

        # 5. Hybrid Reranking or Image-only ranking
        try:
            reranked = self.reranker.rerank(
                candidates=raw_candidates,
                query_text=query_text,
                query_text_vector=query_text_vec,
                query_attributes=query_attributes,
                image_only=(not use_hybrid_reranking),
            )
        except Exception as e:
            logger.exception("Error during reranking.")
            return IdentificationResult(
                status="error",
                overall_confidence=settings.label_no_reliable_match,
                guidance_message="Failed to rerank product candidates.",
                candidates=[],
                disclaimer=settings.repair_part_disclaimer,
                query_preview_available=True,
                error_message=str(e),
            )

        top_candidates = reranked[:top_k]
        top_1 = top_candidates[0]
        overall_confidence = top_1.uncertainty_label

        # 6. Formulate actionable guidance message based on confidence
        if overall_confidence == settings.label_likely_match:
            guidance = (
                f"Found likely match: {top_1.candidate.model_number}. "
                "Please verify visual details with the photos below."
            )
            status = "success"
        elif overall_confidence == settings.label_needs_confirmation:
            guidance = (
                "Candidate models found, but confidence requires user confirmation. "
                "Check the spout curve, handle count, and mounting style, or provide an additional angle."
            )
            status = "uncertain"
        else:
            guidance = (
                "No reliable match could be confirmed. "
                "We recommend taking a clearer photo with good lighting showing the faucet base, "
                "handle mechanism, or any stamped markings."
            )
            status = "no_match"

        # Format candidates for UI
        formatted = []
        for rc in top_candidates:
            cand = rc.candidate
            formatted.append({
                "model_id": cand.model_id,
                "model_number": cand.model_number,
                "product_title": cand.product_title,
                "final_score": rc.final_score,
                "image_score": rc.image_score,
                "text_score": rc.text_score,
                "attribute_score": rc.attribute_score,
                "catalog_quality_score": rc.catalog_quality_score,
                "active_weights": rc.active_weights,
                "evidence": rc.evidence,
                "uncertainty_label": rc.uncertainty_label,
                "best_image_path": cand.best_image_path,
                "image_url": cand.image_url,
                "product_url": cand.product_url,
                "category": cand.category,
                "finish": cand.finish,
                "handle_configuration": cand.handle_configuration,
                "mounting_type": cand.mounting_type,
                "spout_type": cand.spout_type,
                "inventory_status": cand.inventory_status or "Unknown (Real feed not connected)",
            })

        return IdentificationResult(
            status=status,
            overall_confidence=overall_confidence,
            guidance_message=guidance,
            candidates=formatted,
            disclaimer=settings.repair_part_disclaimer,
            query_preview_available=True,
        )

