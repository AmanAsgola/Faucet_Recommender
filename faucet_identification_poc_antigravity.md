# Delta Faucet Identification POC — Build Specification for Antigravity

## 1. Project goal

Build a working proof of concept (POC) that accepts a customer-uploaded faucet photo, searches an authorized Delta Faucet catalog dataset, and returns the three most likely matching faucet models with evidence and uncertainty.

If the exact model is unavailable, recommend visually and specification-similar alternatives only when the relevant catalog/inventory data supports the claim.

**Do not build Neural Collaborative Filtering (NCF) in this POC.** There is no customer–product interaction history. Start with image retrieval and content-based hybrid ranking.

## 2. Important constraints

- Do not scrape or bulk-download Delta Faucet website content unless the data source and use are authorized. Prefer an official feed/API, approved export, or written permission.
- Do not invent model numbers, specifications, inventory status, or compatibility claims.
- Do not describe raw cosine similarity as an accuracy percentage or probability.
- Clearly distinguish `likely_match`, `needs_confirmation`, and `no_reliable_match`.
- Visual similarity does not prove repair-part compatibility.
- Process customer uploads in memory by default; do not persist them unless there is a clear need and an approved retention policy.
- Keep secrets and private catalog assets out of source control.

## 3. Recommended stack

- Python 3.11+
- Streamlit for a simple upload/results UI
- Pillow for image decoding and validation
- OpenCLIP for pretrained image embeddings
- FAISS for a local vector index
- Pandas for catalog CSV ingestion and evaluation
- SQLite for the initial product metadata store; keep the data-access layer replaceable so PostgreSQL can be used later
- Pytest for basic tests

Use CPU inference first. Keep embedding-model initialization outside request handlers so the model is loaded once. Pin package versions after a clean installation succeeds.

## 4. MVP user flow

1. User uploads a JPEG, PNG, or WebP image.
2. Validate file size, file format, decoding, dimensions, and orientation.
3. Show a preview of the uploaded image.
4. Generate a normalized image embedding.
5. Search the catalog image index for the top 20–50 nearest image vectors.
6. Group results by canonical model ID so multiple images of one model do not crowd out other models.
7. Aggregate each model's best image score and fetch its metadata.
8. Return the top three candidates with model number, product title, product image thumbnail, category/finish if available, source link, ranking score, evidence, and uncertainty label.
9. Provide feedback buttons: “Correct match”, “Wrong match”, “Need another photo”.
10. If no reliable match is found, explain that and suggest a clearer photo or a view showing the faucet base/handles/model label.

## 5. Data inputs and schema

Create `data/products.csv` from an authorized catalog source.

Required columns:
- `model_id`: stable canonical internal identifier
- `model_number`: actual catalog model number
- `product_title`
- `image_url` or `local_path`
- `product_url`

Optional columns:
- `category`
- `finish`
- `handle_configuration`
- `mounting_type`
- `spout_type`
- `specifications_json`
- `inventory_status` (only when sourced from a real inventory feed)
- `source_name`
- `source_checked_at`

Use a separate image manifest with one row per image:
- `image_id`
- `model_id`
- `image_url`
- `local_path`
- `view_type`
- `content_hash`
- `download_status`

One model can have multiple images. All images must map to the correct canonical model ID. Preserve original model numbers while storing normalized lookup keys separately.

## 6. Project structure

Create this structure:

```text
faucet-poc/
  app/
    __init__.py
    config.py
    catalog.py
    image_utils.py
    embeddings.py
    index_store.py
    retrieval.py
    rerank.py
    service.py
    ui.py
  scripts/
    validate_catalog.py
    build_index.py
    evaluate.py
  data/
    raw/                 # ignored by git; approved assets only
    processed/
    evaluation/
  tests/
    test_image_utils.py
    test_catalog.py
    test_rerank.py
  .gitignore
  requirements.txt
  .env.example
  README.md
```

## 7. Implementation requirements

### A. Catalog validation

Implement a script that loads the CSV, checks required columns, rejects blank model IDs/model numbers, reports duplicate IDs and suspicious duplicate model numbers, checks local image paths, reports missing metadata, writes a validation report, and does not crawl the website or fetch remote images automatically.

### B. Image validation

Implement a safe function that enforces a configurable maximum upload size (default 10 MB), allows JPEG/PNG/WebP only, verifies actual image content, handles EXIF orientation, converts to RGB, rejects corrupt or extremely small images, handles Pillow decompression-bomb protections, and returns clear user-safe errors.

### C. Embedding model

Create an `ImageEmbedder` class that loads a configured OpenCLIP model/checkpoint once, uses official preprocessing, runs in inference mode, L2-normalizes vectors, and produces consistent float32 vectors for catalog and query images. Record model/checkpoint/package version/embedding dimension in an index manifest. Rebuild the index if the model or preprocessing changes.

### D. Build vector index

Implement `scripts/build_index.py` that reads the validated local image manifest, embeds every valid catalog image, stores vectors in a FAISS inner-product index using normalized vectors, and stores a parallel metadata file mapping each vector position to image ID, model ID, model number, local path, and product URL. Write a manifest with build date, catalog version, model/checkpoint, vector count, and failures. Skip invalid images with logged reasons, fail safely if there are zero valid images, and save outputs atomically where practical.

### E. Retrieval

Embed a customer query image, search the vector index for configurable `top_k_images` (default 50), group image hits by model ID, use the highest image similarity as the initial model-level score, return distinct models sorted by score, and handle empty indexes or invalid vectors gracefully.

### F. Hybrid reranking

Implement image-only ranking first. Then add an explicit reranking interface that can use:
- `image_similarity`
- `text_similarity` if a compatible text-embedding signal is available
- `attribute_match` only when the query attribute is reliably known
- `catalog_quality` only for data completeness, not as evidence that a product is visually correct

Use an initial configurable weighted score, marked unvalidated. Example: `score = 0.60 * image + 0.20 * text + 0.15 * attributes + 0.05 * catalog_quality`.

Do not compare raw scores from unrelated models without calibration. If a feature is missing, renormalize available weights. Unknown attributes must not count as mismatches. Test missing features and tie-breaking. Do not claim hybrid ranking is complete until at least two independent signals are implemented and tested.

### G. Confidence and abstention

Implement configurable thresholds for minimum top score, margin between ranks 1 and 2, and minimum usable evidence. Mark thresholds “not calibrated” until evaluation data is used. Use conservative labels: `likely_match`, `needs_confirmation`, `no_reliable_match`. Do not convert cosine similarity into a percentage probability.

### H. Streamlit UI

Build a compact UI with title “Faucet Finder — POC”, upload and image preview, “Find matching models” button, loading state and safe errors, three candidate cards with image/model/title/available attributes/score/evidence/source link, uncertainty message, feedback controls, and the notice: “Visual candidates only. Confirm the model number before ordering repair parts.” Never fabricate unavailable fields or imply inventory status without real data.

### I. Evaluation

Create an evaluation CSV with `query_image_path`, `true_model_id`, `is_in_catalog`, `image_quality`, `category`, and `notes`.

Report top-1 accuracy, top-3 hit rate, mean reciprocal rank, unknown/no-match false-accept rate where labels exist, coverage vs accuracy for thresholds, latency, and category/quality breakdowns where sample size permits. Do not evaluate only on catalog hero images used to build the index. Use separate real-world query photos where possible and avoid near-duplicate leakage. Do not claim accuracy without measurements.

### J. Tests

Test invalid formats, corrupt bytes, oversized/tiny images, missing CSV columns, duplicate IDs, empty index, multiple image hits for one model, ranking/ties, missing reranking features, uncertain/no-match decisions, and error handling.

## 8. Alternative recommendations — later phase

Do not recommend alternatives based on visual similarity alone.

When the exact model is unavailable:
1. Verify availability using an actual inventory feed. Without one, show availability as “unknown”.
2. Identify hard requirements from trusted catalog data.
3. Exclude candidates that violate verified mandatory compatibility requirements.
4. Rank remaining candidates by visual and attribute similarity.
5. Label alternatives as `visually_similar`, `specification_similar`, or `compatibility_verified` only when authoritative compatibility data supports that claim.
6. Never claim a repair part fits solely because two faucets look alike.

For the first POC, implement the ranking interface and metadata fields, but do not fabricate inventory or compatibility data.

## 9. NCF decision

Do not implement Neural Collaborative Filtering now. The available data is catalog content, not user–item interaction history. Revisit NCF only after meaningful searches, confirmed selections, clicks, purchases, or feedback have been collected. Until then, content-based recommendations are the correct baseline.

## 10. Acceptance criteria

The POC is complete when:
- The app runs locally using documented commands.
- A catalog CSV can be validated reproducibly.
- An index can be built from local, approved images.
- A customer image returns up to three distinct model candidates.
- Every result maps to a real catalog record and source.
- Invalid uploads are handled safely.
- Uncertain queries can be rejected or flagged.
- The UI never fabricates product details, availability, or compatibility.
- Evaluation metrics are generated from labeled examples.
- Tests pass and setup instructions are documented.

## 11. Developer workflow for Antigravity

1. Inspect the repository and summarize its current state before changing files.
2. Create the project structure and configuration.
3. Implement catalog validation and image validation first.
4. Implement embedding and index building.
5. Implement retrieval and tests.
6. Build the Streamlit UI.
7. Add hybrid reranking only after image-only retrieval works.
8. Add evaluation and conservative abstention.
9. Run tests and a smoke test; fix errors.
10. Update README with setup, data format, index rebuild, run commands, limitations, and screenshots if possible.

Do not stop after generating placeholder files. Implement a runnable local POC using a tiny synthetic/demo manifest for tests, but never invent Delta product records. If no authorized real catalog data is available, clearly document that real matching cannot be validated until the user supplies it.

## 12. Expected commands

Document and make these work where applicable:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python scripts/validate_catalog.py
python scripts/build_index.py
streamlit run app/ui.py
pytest -q
python scripts/evaluate.py
```

If a command requires catalog data or an index that does not exist yet, provide a clear actionable error message rather than a traceback.

## 13. Final output requested from Antigravity

After implementation, provide:
- Summary of files created/changed
- Exact local setup/run commands
- Required CSV schema and one clearly marked dummy example row
- Tests run and results
- What is functional vs still a placeholder
- Known limitations
- Next steps to connect the approved Delta catalog and real inventory feed
