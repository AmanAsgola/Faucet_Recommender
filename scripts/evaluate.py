"""Evaluation script to benchmark retrieval accuracy, MRR, abstention, and latency."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from app.config import settings
from app.service import FaucetIdentificationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Faucet Identification accuracy and latency.")
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=settings.evaluation_dir / "queries.csv",
        help="Path to evaluation queries CSV (default: data/evaluation/queries.csv)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=settings.processed_dir / "evaluation_report.json",
        help="Path to output evaluation report JSON",
    )
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        default=True,
        help="Use hybrid reranking during evaluation",
    )

    args = parser.parse_args()

    print("==================================================")
    print(" Delta Faucet POC — Accuracy & Latency Evaluation")
    print("==================================================")

    if not args.eval_csv.is_file():
        print(f"\n[ERROR] Evaluation dataset not found at: {args.eval_csv}")
        print("Please provide an evaluation CSV with columns:")
        print("query_image_path, true_model_id, is_in_catalog, image_quality, category, notes")
        return 1

    try:
        eval_df = pd.read_csv(args.eval_csv)
    except Exception as e:
        print(f"[ERROR] Failed to read evaluation CSV: {e}")
        return 1

    print(f"Loaded {len(eval_df)} evaluation test queries.")

    service = FaucetIdentificationService()
    ready, msg = service.check_system_ready()
    if not ready:
        print(f"\n[ERROR] System is not ready for evaluation: {msg}")
        return 1

    # Preload embedder to avoid counting warm-up time in latency
    print("Pre-warming model...")
    service._get_embedder()

    # Track metrics
    total_queries = len(eval_df)
    in_catalog_queries = 0
    out_catalog_queries = 0

    top_1_hits = 0
    top_3_hits = 0
    reciprocal_ranks = []
    latencies = []

    false_accepts = 0  # Out of catalog queries accepted as "likely_match"
    true_rejects = 0   # Out of catalog queries abstained

    # Breakdown by image quality
    quality_stats: Dict[str, Dict[str, int]] = {}

    print("\nRunning evaluation on test cases...")
    for idx, row in eval_df.iterrows():
        img_rel_path = row["query_image_path"]
        img_path = Path(img_rel_path)
        if not img_path.is_absolute():
            img_path = BASE_DIR / img_path

        true_model = str(row["true_model_id"]).strip() if pd.notna(row.get("true_model_id")) else ""
        is_in_cat = bool(row.get("is_in_catalog", True))
        quality = str(row.get("image_quality", "unknown"))

        if quality not in quality_stats:
            quality_stats[quality] = {"total": 0, "top_1": 0, "top_3": 0}
        quality_stats[quality]["total"] += 1

        t0 = time.perf_counter()
        res = service.identify(
            image_input=img_path,
            use_hybrid_reranking=args.use_hybrid,
            top_k=3,
        )
        t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(t_elapsed)

        retrieved_ids = [c["model_id"] for c in res.candidates]

        if is_in_cat and true_model:
            in_catalog_queries += 1
            # Check rank
            rank = None
            for r_idx, c_id in enumerate(retrieved_ids, start=1):
                if c_id == true_model:
                    rank = r_idx
                    break

            if rank == 1:
                top_1_hits += 1
                quality_stats[quality]["top_1"] += 1
            if rank and rank <= 3:
                top_3_hits += 1
                quality_stats[quality]["top_3"] += 1

            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        else:
            out_catalog_queries += 1
            if res.overall_confidence == settings.label_likely_match:
                false_accepts += 1
            else:
                true_rejects += 1

    # Calculate aggregate metrics
    top_1_acc = (top_1_hits / in_catalog_queries) if in_catalog_queries > 0 else 0.0
    top_3_acc = (top_3_hits / in_catalog_queries) if in_catalog_queries > 0 else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    false_accept_rate = (false_accepts / out_catalog_queries) if out_catalog_queries > 0 else 0.0

    print("\n---------------- Evaluation Results ----------------")
    print(f"Total Evaluated Queries: {total_queries}")
    print(f"In-Catalog Queries:      {in_catalog_queries}")
    print(f"Out-of-Catalog Queries:  {out_catalog_queries}")
    print("----------------------------------------------------")
    print(f"Top-1 Accuracy:          {top_1_acc:.2%}")
    print(f"Top-3 Hit Rate:          {top_3_acc:.2%}")
    print(f"Mean Reciprocal Rank:    {mrr:.4f}")
    if out_catalog_queries > 0:
        print(f"False Accept Rate (OOC): {false_accept_rate:.2%} ({false_accepts}/{out_catalog_queries})")
    print("----------------------------------------------------")
    print(f"Latency Mean:            {mean_lat:.1f} ms")
    print(f"Latency P50 (median):    {p50_lat:.1f} ms")
    print(f"Latency P95:             {p95_lat:.1f} ms")

    print("\nBreakdown by Image Quality:")
    for q, st in quality_stats.items():
        if st["total"] > 0:
            print(f"   • {q.title()}: Top-1={st['top_1']/st['total']:.1%}, Top-3={st['top_3']/st['total']:.1%} (N={st['total']})")

    report_data = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_queries": total_queries,
        "in_catalog_queries": in_catalog_queries,
        "out_catalog_queries": out_catalog_queries,
        "metrics": {
            "top_1_accuracy": round(top_1_acc, 4),
            "top_3_hit_rate": round(top_3_acc, 4),
            "mean_reciprocal_rank": round(mrr, 4),
            "false_accept_rate": round(false_accept_rate, 4),
            "latency_ms_mean": round(mean_lat, 2),
            "latency_ms_p50": round(p50_lat, 2),
            "latency_ms_p95": round(p95_lat, 2),
        },
        "quality_breakdown": quality_stats,
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nReport written to: {args.output_report}")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

