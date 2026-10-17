"""Unit tests for catalog validation, normalization, and SQLite metadata store."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from app.catalog import (
    CatalogStore,
    normalize_model_key,
    validate_catalog_df,
    validate_image_manifest_df,
)


def test_normalize_model_key():
    assert normalize_model_key("9178-DST") == "9178DST"
    assert normalize_model_key("  9159-bl-dst  ") == "9159BLDST"
    assert normalize_model_key("model_123#x") == "MODEL123X"


def test_catalog_validation_missing_columns():
    df = pd.DataFrame([{"model_id": "M1", "product_title": "Faucet"}])
    report = validate_catalog_df(df)
    assert not report.is_valid
    assert any("Missing required columns" in e for e in report.errors)


def test_catalog_validation_blank_ids():
    df = pd.DataFrame([
        {
            "model_id": "",
            "model_number": "123",
            "product_title": "Test Faucet",
            "product_url": "https://example.com/1",
            "local_path": "path.png",
        }
    ])
    report = validate_catalog_df(df)
    assert not report.is_valid
    assert any("blank or missing 'model_id'" in e for e in report.errors)


def test_catalog_validation_duplicate_ids():
    df = pd.DataFrame([
        {
            "model_id": "M-DUP",
            "model_number": "123",
            "product_title": "Test 1",
            "product_url": "https://example.com/1",
            "local_path": "path1.png",
        },
        {
            "model_id": "M-DUP",
            "model_number": "456",
            "product_title": "Test 2",
            "product_url": "https://example.com/2",
            "local_path": "path2.png",
        },
    ])
    report = validate_catalog_df(df)
    assert not report.is_valid
    assert any("duplicate model_id" in e for e in report.errors)


def test_catalog_store_sync_and_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_catalog.db"
        store = CatalogStore(db_path)

        df = pd.DataFrame([
            {
                "model_id": "M-001",
                "model_number": "9178-DST",
                "product_title": "Leland Faucet",
                "product_url": "https://example.com/leland",
                "category": "Kitchen",
                "finish": "Chrome",
            }
        ])

        count = store.sync_from_dataframes(df)
        assert count == 1
        assert store.count_products() == 1

        # Query by model_id
        p = store.get_product("M-001")
        assert p is not None
        assert p["model_number"] == "9178-DST"
        assert p["category"] == "Kitchen"

        # Query by model_number (normalized)
        p_by_num = store.get_product_by_model_number("9178dst")
        assert p_by_num is not None
        assert p_by_num["model_id"] == "M-001"

