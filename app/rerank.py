"""Hybrid reranking combining visual similarity, text/metadata matching, and quality signals."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.retrieval import RetrievedCandidate

logger = logging.getLogger(__name__)


@dataclass
class RerankedCandidate:
    candidate: RetrievedCandidate
    final_score: float
    image_score: float
    text_score: Optional[float]
    attribute_score: Optional[float]
    catalog_quality_score: float
    active_weights: Dict[str, float]
    evidence: List[str]
    uncertainty_label: str  # likely_match, needs_confirmation, no_reliable_match
    is_calibrated: bool = False  # Explicitly marked uncalibrated POC baseline


def compute_catalog_quality(candidate: RetrievedCandidate) -> float:
    """Compute data completeness score for catalog record (0.0 to 1.0).
    
    Note: As specified, data completeness is only for catalog quality, NOT
    as evidence that a product is visually correct.
    """
    fields_to_check = [
        candidate.product_title,
        candidate.product_url,
        candidate.category,
        candidate.finish,
        candidate.handle_configuration,
        candidate.mounting_type,
        candidate.spout_type,
        candidate.best_image_path or candidate.image_url,
    ]
    present = sum(1 for f in fields_to_check if f is not None and str(f).strip() != "")
    return float(present) / len(fields_to_check)


def compute_attribute_match(
    candidate: RetrievedCandidate,
    query_attributes: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[float], List[str]]:
    """Compute attribute match score between query attributes and candidate attributes.
    
    Returns:
        (score, evidence_items)
        If query_attributes is None or empty, returns (None, []) so weight is renormalized.
    """
    if not query_attributes:
        return None, []

    matched = 0
    total_specified = 0
    evidence = []

    attr_map = {
        "category": candidate.category,
        "finish": candidate.finish,
        "handle_configuration": candidate.handle_configuration,
        "mounting_type": candidate.mounting_type,
        "spout_type": candidate.spout_type,
    }

    for key, q_val in query_attributes.items():
        if not q_val or str(q_val).strip() == "":
            continue
        cand_val = attr_map.get(key)
        total_specified += 1
        if cand_val is not None:
            # Case-insensitive comparison
            if str(q_val).strip().lower() == str(cand_val).strip().lower():
                matched += 1
                evidence.append(f"Attribute match ({key}): {cand_val}")
            else:
                evidence.append(f"Attribute mismatch ({key}): query '{q_val}' vs '{cand_val}'")
        else:
            # Unknown attribute in catalog: do NOT penalize as mismatch
            # Count it as neutral (0.5) or exclude from denominator
            total_specified -= 1
            evidence.append(f"Attribute unknown in catalog ({key})")

    if total_specified == 0:
        return None, evidence

    score = float(matched) / total_specified
    return score, evidence


def compute_text_similarity(
    candidate: RetrievedCandidate,
    query_text_vector: Optional[np.ndarray] = None,
    candidate_text_vector: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], List[str]]:
    """Compute cosine similarity between query text embedding and candidate text embedding."""
    if query_text_vector is None or candidate_text_vector is None:
        return None, []

    sim = float(np.dot(query_text_vector, candidate_text_vector))
    # OpenCLIP cosine similarity is in [-1, 1], clip to [0, 1]
    sim = max(0.0, min(1.0, sim))
    return sim, [f"Text similarity: {sim:.3f}"]


class HybridReranker:
    """Reranker supporting image-only baseline and multi-signal hybrid reranking."""

    def __init__(
        self,
        weight_image: Optional[float] = None,
        weight_text: Optional[float] = None,
        weight_attributes: Optional[float] = None,
        weight_catalog_quality: Optional[float] = None,
        high_threshold: Optional[float] = None,
        low_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
    ):
        self.w_image = weight_image if weight_image is not None else settings.weight_image
        self.w_text = weight_text if weight_text is not None else settings.weight_text
        self.w_attr = weight_attributes if weight_attributes is not None else settings.weight_attributes
        self.w_quality = weight_catalog_quality if weight_catalog_quality is not None else settings.weight_catalog_quality

        self.high_threshold = high_threshold or settings.similarity_threshold_high
        self.low_threshold = low_threshold or settings.similarity_threshold_low
        self.margin_threshold = margin_threshold or settings.margin_threshold

    def rerank(
        self,
        candidates: List[RetrievedCandidate],
        query_text: Optional[str] = None,
        query_text_vector: Optional[np.ndarray] = None,
        candidate_text_vectors: Optional[Dict[str, np.ndarray]] = None,
        query_attributes: Optional[Dict[str, str]] = None,
        image_only: bool = False,
    ) -> List[RerankedCandidate]:
        """Rerank retrieved candidates using weighted signals and assign uncertainty labels.
        
        Args:
            candidates: List of RetrievedCandidate objects from retrieval stage.
            query_text: Optional text query (e.g. user description or finish).
            query_text_vector: Optional OpenCLIP text vector for query.
            candidate_text_vectors: Map of model_id -> text embedding.
            query_attributes: Optional user-specified attributes (category, finish, etc.).
            image_only: If True, bypass hybrid reranking and use purely image similarity.

        Returns:
            List of RerankedCandidate sorted by final_score descending.
        """
        if not candidates:
            return []

        results: List[RerankedCandidate] = []

        for cand in candidates:
            evidence: List[str] = [f"Image similarity: {cand.image_similarity:.3f}"]
            img_score = max(0.0, min(1.0, cand.image_similarity))

            if image_only:
                results.append(
                    RerankedCandidate(
                        candidate=cand,
                        final_score=round(img_score, 4),
                        image_score=round(img_score, 4),
                        text_score=None,
                        attribute_score=None,
                        catalog_quality_score=0.0,
                        active_weights={"image": 1.0},
                        evidence=evidence,
                        uncertainty_label="unassigned",
                        is_calibrated=False,
                    )
                )
                continue

            # Hybrid mode: collect available signals
            active_weights: Dict[str, float] = {"image": self.w_image}
            scores_dict: Dict[str, float] = {"image": img_score}

            # 1. Text signal
            cand_text_vec = candidate_text_vectors.get(cand.model_id) if candidate_text_vectors else None
            txt_score, txt_evidence = compute_text_similarity(cand, query_text_vector, cand_text_vec)
            if txt_score is not None:
                scores_dict["text"] = txt_score
                active_weights["text"] = self.w_text
                evidence.extend(txt_evidence)

            # 2. Attribute signal
            attr_score, attr_evidence = compute_attribute_match(cand, query_attributes)
            if attr_score is not None:
                scores_dict["attributes"] = attr_score
                active_weights["attributes"] = self.w_attr
                evidence.extend(attr_evidence)

            # 3. Catalog quality (completeness)
            quality_score = compute_catalog_quality(cand)
            scores_dict["catalog_quality"] = quality_score
            active_weights["catalog_quality"] = self.w_quality
            evidence.append(f"Catalog completeness: {quality_score:.0%}")

            # Renormalize active weights to sum to 1.0
            total_weight = sum(active_weights.values())
            normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

            # Calculate composite final score
            final_score = sum(scores_dict[k] * normalized_weights[k] for k in scores_dict)

            results.append(
                RerankedCandidate(
                    candidate=cand,
                    final_score=round(final_score, 4),
                    image_score=round(img_score, 4),
                    text_score=round(txt_score, 4) if txt_score is not None else None,
                    attribute_score=round(attr_score, 4) if attr_score is not None else None,
                    catalog_quality_score=round(quality_score, 4),
                    active_weights=normalized_weights,
                    evidence=evidence,
                    uncertainty_label="unassigned",
                    is_calibrated=False,
                )
            )

        # Sort by final score descending (with image similarity as tie-breaker)
        results.sort(key=lambda x: (x.final_score, x.image_score), reverse=True)

        # Assign conservative uncertainty labels
        top_1_score = results[0].final_score
        top_1_img = results[0].image_score
        top_2_score = results[1].final_score if len(results) > 1 else 0.0
        margin = top_1_score - top_2_score

        for i, res in enumerate(results):
            if i == 0:
                if (
                    top_1_score >= self.high_threshold
                    and top_1_img >= self.high_threshold
                    and (len(results) == 1 or margin >= self.margin_threshold)
                ):
                    res.uncertainty_label = settings.label_likely_match
                elif top_1_score >= self.low_threshold and top_1_img >= self.low_threshold:
                    res.uncertainty_label = settings.label_needs_confirmation
                else:
                    res.uncertainty_label = settings.label_no_reliable_match
            else:
                # Rank 2 and 3
                if res.final_score >= self.low_threshold and res.image_score >= self.low_threshold:
                    res.uncertainty_label = settings.label_needs_confirmation
                else:
                    res.uncertainty_label = settings.label_no_reliable_match

        return results

