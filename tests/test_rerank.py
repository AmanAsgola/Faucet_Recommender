"""Unit tests for hybrid reranking, feature weights, and uncertainty classification."""

import numpy as np
import pytest

from app.config import settings
from app.rerank import (
    HybridReranker,
    compute_attribute_match,
    compute_catalog_quality,
)
from app.retrieval import RetrievedCandidate


def create_candidate(
    model_id="M1",
    model_number="MOD-1",
    similarity=0.80,
    category="Kitchen",
    finish="Chrome",
) -> RetrievedCandidate:
    return RetrievedCandidate(
        model_id=model_id,
        model_number=model_number,
        product_title=f"Title {model_id}",
        image_similarity=similarity,
        best_image_id=f"img_{model_id}",
        product_url="https://example.com",
        best_image_path="test.png",
        category=category,
        finish=finish,
        handle_configuration="Single Handle",
        mounting_type="Deck Mount",
        spout_type="High Arc",
    )


def test_compute_catalog_quality():
    cand_full = create_candidate()
    q_score = compute_catalog_quality(cand_full)
    assert q_score == 1.0

    cand_partial = RetrievedCandidate(
        model_id="M2",
        model_number="MOD-2",
        product_title="Title M2",
        image_similarity=0.70,
        best_image_id="img_M2",
        # missing attributes
    )
    q_partial = compute_catalog_quality(cand_partial)
    assert 0.0 < q_partial < 1.0


def test_compute_attribute_match_matching():
    cand = create_candidate(category="Kitchen", finish="Chrome")
    query_attrs = {"category": "Kitchen", "finish": "Chrome"}
    score, evidence = compute_attribute_match(cand, query_attrs)
    assert score == 1.0
    assert len(evidence) == 2


def test_compute_attribute_match_unknown_attribute_not_penalized():
    # If catalog attribute is None, it should not penalize as mismatch
    cand = create_candidate(finish=None)
    cand.finish = None
    query_attrs = {"category": "Kitchen", "finish": "Matte Black"}
    score, evidence = compute_attribute_match(cand, query_attrs)
    # category matches (1/1 evaluated), finish was unknown in catalog
    assert score == 1.0
    assert any("unknown in catalog" in e for e in evidence)


def test_image_only_mode():
    c1 = create_candidate("M1", similarity=0.85)
    c2 = create_candidate("M2", similarity=0.75)
    reranker = HybridReranker()
    results = reranker.rerank([c1, c2], image_only=True)

    assert len(results) == 2
    assert results[0].final_score == 0.85
    assert results[0].candidate.model_id == "M1"
    assert results[0].active_weights == {"image": 1.0}


def test_hybrid_weight_renormalization_when_text_missing():
    c1 = create_candidate("M1", similarity=0.80, category="Kitchen")
    reranker = HybridReranker(
        weight_image=0.60, weight_text=0.20, weight_attributes=0.15, weight_catalog_quality=0.05
    )
    # No text query, but attribute query provided
    results = reranker.rerank([c1], query_attributes={"category": "Kitchen"}, image_only=False)
    assert len(results) == 1
    # text is missing, so weights should renormalize over (image=0.60, attr=0.15, quality=0.05) -> sum = 0.80
    weights = results[0].active_weights
    assert "text" not in weights
    assert pytest.approx(sum(weights.values()), 0.001) == 1.0
    assert pytest.approx(weights["image"], 0.001) == 0.60 / 0.80


def test_conservative_uncertainty_labels():
    reranker = HybridReranker(high_threshold=0.70, low_threshold=0.50, margin_threshold=0.05)

    # 1. High score with clear margin -> likely_match
    c1 = create_candidate("M1", similarity=0.85)
    c2 = create_candidate("M2", similarity=0.60)
    res = reranker.rerank([c1, c2], image_only=True)
    assert res[0].uncertainty_label == settings.label_likely_match
    assert res[1].uncertainty_label == settings.label_needs_confirmation

    # 2. Moderate score -> needs_confirmation
    c3 = create_candidate("M3", similarity=0.58)
    res2 = reranker.rerank([c3], image_only=True)
    assert res2[0].uncertainty_label == settings.label_needs_confirmation

    # 3. Low score -> no_reliable_match
    c4 = create_candidate("M4", similarity=0.35)
    res3 = reranker.rerank([c4], image_only=True)
    assert res3[0].uncertainty_label == settings.label_no_reliable_match

