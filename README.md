# Delta Faucet Identification POC

A visual search and identification proof of concept (POC) that matches customer-uploaded faucet photos against an authorized Delta Faucet catalog using OpenCLIP visual embeddings, FAISS vector indexing, and multi-signal hybrid reranking.

---

## 1. System Architecture & Features

- **Safe Image Ingestion (`app/image_utils.py`)**: Validates uploaded files (JPEG, PNG, WebP), checks file size limits (default 10 MB), applies EXIF orientation transpose, converts to standardized RGB, and rejects corrupt or undersized images.
- **OpenCLIP Embeddings (`app/embeddings.py`)**: Uses pretrained `ViT-B-32` (`openai` weights) running on CPU to generate 512-dimensional L2-normalized feature vectors. Embedder is cached outside Streamlit request loops.
- **FAISS Inner-Product Vector Index (`app/index_store.py`)**: Fast cosine similarity retrieval over normalized vector embeddings with parallel metadata mapping and atomic persistence.
- **Model-Level Candidate Aggregation (`app/retrieval.py`)**: Retrieves top-k image hits and groups them by canonical model ID (`max(image_similarity)`) so multiple photos of a single model do not crowd out other models.
- **Hybrid Reranking (`app/rerank.py`)**: Combines visual similarity with query text matching, user-specified attributes (category, finish), and catalog completeness. Automatically renormalizes weights when features are missing.
- **Conservative Uncertainty Handling (`app/service.py`)**: Classifies results into `likely_match`, `needs_confirmation`, and `no_reliable_match`. Never presents geometric cosine similarity as an accuracy percentage or probability.
- **Streamlit Interface (`app/ui.py`)**: Clean web UI showing uploaded image preview, top 3 candidate cards, image thumbnails, specification breakdown, evidence metrics, user feedback buttons, and the mandatory repair-part disclaimer.

---

## 2. Environment Setup

Tested and verified on **Python 3.13** (compatible with Python 3.11+).

### Windows PowerShell

```powershell
# 1. Create local virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# (If activation script execution is restricted on Windows):
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
# 1. Create local virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 3. Data Schema & Approved Catalog Assets

> [!IMPORTANT]
> **Data Compliance:** Do not scrape or bulk-download Delta Faucet website content without authorization. This POC uses official product feeds or clearly marked synthetic test fixtures.

### A. Products Catalog CSV (`data/products.csv`)

| Column Name | Type | Description | Required? | Example |
| :--- | :--- | :--- | :--- | :--- |
| `model_id` | string | Canonical unique product identifier | **Yes** | `DELTA-9178-DST` |
| `model_number` | string | Catalog model number | **Yes** | `9178-DST` |
| `product_title` | string | Full commercial product title | **Yes** | `Leland Single Handle Pull-Down Kitchen Faucet` |
| `product_url` | string | Official product documentation URL | **Yes** | `https://www.deltafaucet.com/product/9178-DST` |
| `local_path` | string | Path to primary catalog image file | Optional* | `data/raw/images/9178-DST.png` |
| `image_url` | string | Remote asset image URL | Optional* | `https://media.deltafaucet.com/9178-DST.jpg` |
| `category` | string | Product category (e.g. Kitchen, Bath) | Optional | `Kitchen` |
| `finish` | string | Finish name | Optional | `Chrome` |
| `handle_configuration`| string | Handle count / style | Optional | `Single Handle` |
| `mounting_type` | string | Deck Mount, Single Hole, Wall Mount | Optional | `Deck Mount` |
| `spout_type` | string | Spout curve / mechanism | Optional | `High Arc Pull-Down` |
| `specifications_json` | string | JSON blob of additional specs | Optional | `{"flow_rate": "1.8 gpm"}` |
| `inventory_status` | string | Live feed status (only from inventory API) | Optional | `In Stock` |
| `source_name` | string | Feed or export name | Optional | `Authorized Product Export v2` |
| `source_checked_at` | string | ISO timestamp of source verification | Optional | `2026-10-17T00:00:00Z` |

*\* At least one image locator (`local_path` or `image_url`) is required.*

### B. Image Manifest CSV (`data/image_manifest.csv`)

For models with multiple views (e.g., primary, angled, handle detail, diagram):

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `image_id` | string | Unique identifier for this image |
| `model_id` | string | Foreign key referencing `data/products.csv` |
| `local_path` | string | Relative path to image on local disk |
| `view_type` | string | `primary`, `angle`, `handle`, `close_up` |
| `content_hash` | string | SHA256 checksum of image file |
| `download_status` | string | `local_verified` |

---

## 4. Running the POC Workflow

All commands must be executed within the active virtual environment:

### Step 1: Ingest Delta Full-Line Catalogue (or Generate Synthetic Fixtures)

The repository includes the authorized official **Delta Kitchen & Bath Full-Line Catalogue** (`Kitchen & Bath Full-Line Catalogue.pdf`).

To extract all 297 authentic Delta models, specifications, and imagery:
```bash
python scripts/extract_catalogue.py
```
*(Alternatively, to run with synthetic mock fixtures for offline testing: `python scripts/generate_fixtures.py`)*

### Step 2: Validate Catalog and Image Manifest

```bash
python scripts/validate_catalog.py
```
- Validates required schema, primary keys, and checks disk presence of referenced images.
- Synchronizes the 297 valid products into the local SQLite store (`data/processed/faucet_catalog.db`).
- Writes validation report to `data/processed/catalog_validation_report.json`.

### Step 3: Build the FAISS Vector Index

```bash
python scripts/build_index.py
```
- Embeds every catalog image with OpenCLIP (`ViT-B-32`).
- Saves normalized FAISS IndexFlatIP to `data/processed/faucet_index.faiss`.
- Saves parallel metadata to `data/processed/faucet_metadata.json` and build manifest to `data/processed/index_manifest.json`.

### Step 4: Run Automated Unit Tests

```bash
python -m pytest -q
```
Executes the full test suite covering image decoding, corrupt uploads, schema validation, candidate deduplication, hybrid reranking, and confidence thresholds.

### Step 5: Run Evaluation Benchmark

```bash
python scripts/evaluate.py
```
Computes Top-1 accuracy, Top-3 hit rate, Mean Reciprocal Rank (MRR), out-of-catalog false accept rate, and query latency percentiles.

### Step 6: Launch the Streamlit User Interface

```bash
python -m streamlit run app/ui.py
```
Access the application at `http://localhost:8501`.

---

## 5. Decision Rules & Uncertainty Labels

The system assigns conservative confidence labels to prevent false purchase decisions:

| Uncertainty Label | Conditions | Actionable Guidance |
| :--- | :--- | :--- |
| **`likely_match`** | Top score $\ge 0.85$, Top image similarity $\ge 0.85$, and margin to Rank 2 $\ge 0.03$. | High visual confidence. User confirms finish and model number. |
| **`needs_confirmation`** | Top score $\ge 0.80$ but lower margin or visual ambiguities. | Candidate models presented. User prompted to verify spout curve or provide another angle. |
| **`no_reliable_match`** | Top visual score $< 0.80$. | System abstains from guessing. Recommends clearer photo of faucet base, handles, or stamped label. |

> [!WARNING]
> **Mandatory Disclaimer:** "Visual candidates only. Confirm the model number before ordering repair parts. Visual similarity does not prove repair-part compatibility."

---

## 6. Known Limitations & Architectural Decisions

1. **Neural Collaborative Filtering (NCF) Not Implemented**:
   - Deferred intentionally as specified. There is currently no historical customer interaction or clickstream data. Content-based visual + hybrid retrieval is the appropriate baseline.
2. **Inventory Availability**:
   - Without an authorized live inventory API, inventory status defaults to `"Unknown (Real feed not connected)"`. Never fabricated.
3. **Alternative Recommendations**:
   - Visual similarity alone does not establish mechanical fit or plumbing compatibility. Compatibility-verified recommendations require an authoritative parts cross-reference table.
4. **Scraping Prohibition**:
   - No scraping or automated crawling of `deltafaucet.com` is performed.

---

## 7. Connecting Authorized Delta Catalog & Inventory Feeds

When ready to transition from synthetic fixtures to production Delta catalog data:

1. **Export Catalog Data**: Export official Delta product data matching the schema in Section 3 to `data/products.csv`.
2. **Download Approved Product Assets**: Place authorized high-resolution product imagery in `data/raw/images/` and record paths in `data/image_manifest.csv`.
3. **Rebuild the Vector Index**:
   ```bash
   python scripts/validate_catalog.py
   python scripts/build_index.py
   ```
4. **Connect Live Inventory Feed**: Configure an API adapter in `app/catalog.py` to query Delta's ERP/inventory system using the canonical `model_number` or `model_id`.

