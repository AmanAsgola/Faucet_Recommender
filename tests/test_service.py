"""Unit tests for the end-to-end FaucetIdentificationService."""

import pytest
from app.config import settings
from app.service import FaucetIdentificationService, IdentificationResult


def test_service_unready_when_index_missing(tmp_path):
    # Empty stores pointing to non-existent paths
    service = FaucetIdentificationService()
    service.index_store.index_path = tmp_path / "missing.faiss"
    service.index_store.metadata_path = tmp_path / "missing.json"

    result = service.identify(b"dummy")
    assert result.status == "error"
    assert "missing" in result.guidance_message.lower()
    assert result.disclaimer == settings.repair_part_disclaimer


def test_service_handles_corrupt_upload(tmp_path):
    service = FaucetIdentificationService()
    # Mock ready
    service.check_system_ready = lambda: (True, "Ready")

    result = service.identify(b"corrupt image bytes")
    assert result.status == "error"
    assert "validation failed" in result.guidance_message.lower()
    assert result.disclaimer == settings.repair_part_disclaimer

