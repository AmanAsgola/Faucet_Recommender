"""Unit tests for vector retrieval and model-level candidate aggregation."""

import numpy as np
import pytest

from app.catalog import CatalogStore
from app.retrieval import RetrievedCandidate, Retriever


class MockIndexStore:
    """Mock index store for fast unit testing of candidate aggregation logic."""

    def __init__(self, scores, indices, metadata):
        self._scores = np.array(scores, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int64)
        self.metadata = metadata
        self.manifest = {"vector_count": len(metadata)}

    def load(self):
        # Create a mock index object
        class MockFaissIndex:
            ntotal = len(self._indices)
        return MockFaissIndex(), self.metadata, self.manifest

    def search(self, index, query_vector, top_k=50):
        return self._scores[:top_k], self._indices[:top_k]


def test_retriever_model_aggregation_and_deduplication():
    # Simulate an index with 4 image hits:
    # 2 hits for MODEL-A (scores 0.82 and 0.90)
    # 1 hit for MODEL-B (score 0.75)
    # 1 hit for MODEL-C (score 0.85)
    metadata = [
        {"model_id": "MODEL-A", "model_number": "MA-1", "image_id": "img_a1", "local_path": "a1.png"},
        {"model_id": "MODEL-A", "model_number": "MA-1", "image_id": "img_a2", "local_path": "a2.png"},
        {"model_id": "MODEL-B", "model_number": "MB-1", "image_id": "img_b1", "local_path": "b1.png"},
        {"model_id": "MODEL-C", "model_number": "MC-1", "image_id": "img_c1", "local_path": "c1.png"},
    ]
    # Search returns indices in raw order: 1 (score 0.90), 3 (score 0.85), 0 (score 0.82), 2 (score 0.75)
    scores = [0.90, 0.85, 0.82, 0.75]
    indices = [1, 3, 0, 2]

    mock_store = MockIndexStore(scores, indices, metadata)
    retriever = Retriever(index_store=mock_store, catalog_store=CatalogStore())
    retriever.load()

    dummy_query = np.zeros(512, dtype=np.float32)
    candidates = retriever.retrieve(dummy_query)

    # Must return 3 distinct models, NOT 4 raw image hits!
    assert len(candidates) == 3
    # Sorted by best score: MODEL-A (0.90), MODEL-C (0.85), MODEL-B (0.75)
    assert candidates[0].model_id == "MODEL-A"
    assert pytest.approx(candidates[0].image_similarity, 0.001) == 0.90
    assert candidates[0].best_image_id == "img_a2"
    assert candidates[0].image_hits_count == 2

    assert candidates[1].model_id == "MODEL-C"
    assert pytest.approx(candidates[1].image_similarity, 0.001) == 0.85

    assert candidates[2].model_id == "MODEL-B"
    assert pytest.approx(candidates[2].image_similarity, 0.001) == 0.75


def test_empty_index_handling():
    mock_store = MockIndexStore([], [], [])
    retriever = Retriever(index_store=mock_store, catalog_store=CatalogStore())
    retriever.load()

    dummy_query = np.zeros(512, dtype=np.float32)
    candidates = retriever.retrieve(dummy_query)
    assert candidates == []

