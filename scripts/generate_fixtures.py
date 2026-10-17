"""Generate clearly labelled synthetic fixtures for testing and local demonstration."""

import json
from pathlib import Path
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BASE_DIR / "data" / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
EVAL_IMAGES_DIR = BASE_DIR / "data" / "evaluation" / "images"


def draw_synthetic_faucet(
    faucet_type: str,
    finish_color: tuple,
    output_path: Path,
    bg_color=(245, 245, 245),
    width=300,
    height=300,
    variation: str = "standard",
):
    """Draw a stylized synthetic faucet for testing OpenCLIP embeddings."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Countertop line
    draw.rectangle([20, 260, 280, 275], fill=(180, 180, 180))

    if faucet_type == "gooseneck":
        # High arc pull-down kitchen faucet
        # Base
        draw.rectangle([130, 230, 170, 260], fill=finish_color)
        # Vertical riser
        draw.rectangle([140, 120, 160, 230], fill=finish_color)
        # High arc curve
        draw.arc([90, 40, 210, 160], start=180, end=0, fill=finish_color, width=20)
        # Spout head pointing down
        draw.rectangle([80, 100, 100, 140], fill=finish_color)
        # Single lever handle on side
        draw.line([160, 200, 210, 180], fill=finish_color, width=8)

    elif faucet_type == "pull_out":
        # Low arc pull-out kitchen faucet
        draw.rectangle([120, 220, 180, 260], fill=finish_color)
        # Angled spout
        draw.polygon([(135, 220), (165, 220), (90, 140), (70, 150)], fill=finish_color)
        # Pullout spray head
        draw.rectangle([60, 140, 85, 175], fill=finish_color)
        # Top lever
        draw.line([150, 210, 180, 160], fill=finish_color, width=10)

    elif faucet_type == "widespread_bath":
        # Two-handle widespread bathroom faucet
        # Central spout
        draw.rectangle([135, 170, 165, 260], fill=finish_color)
        draw.arc([110, 120, 190, 200], start=210, end=330, fill=finish_color, width=16)
        draw.rectangle([170, 150, 190, 180], fill=finish_color)
        # Left handle
        draw.rectangle([50, 210, 80, 260], fill=finish_color)
        draw.line([40, 210, 90, 210], fill=finish_color, width=8)
        # Right handle
        draw.rectangle([220, 210, 250, 260], fill=finish_color)
        draw.line([210, 210, 260, 210], fill=finish_color, width=8)

    elif faucet_type == "modern_single_hole":
        # Contemporary sharp geometric bathroom faucet
        draw.rectangle([135, 110, 165, 260], fill=finish_color)
        # Sharp 90-degree horizontal spout
        draw.rectangle([70, 110, 140, 130], fill=finish_color)
        # Top joystick handle
        draw.line([150, 110, 150, 60], fill=finish_color, width=8)

    elif faucet_type == "bridge_kitchen":
        # Traditional two-handle bridge faucet
        # Left leg
        draw.rectangle([70, 190, 95, 260], fill=finish_color)
        # Right leg
        draw.rectangle([205, 190, 230, 260], fill=finish_color)
        # Horizontal bridge
        draw.rectangle([70, 180, 230, 200], fill=finish_color)
        # High center spout
        draw.rectangle([138, 100, 162, 180], fill=finish_color)
        draw.arc([100, 40, 200, 140], start=180, end=0, fill=finish_color, width=18)
        draw.rectangle([90, 90, 110, 130], fill=finish_color)

    elif faucet_type == "negative_sample":
        # Non-faucet object for testing out-of-catalog abstention (e.g. geometric pattern / background)
        draw.ellipse([40, 40, 260, 260], fill=(70, 130, 180), outline=(20, 50, 100), width=6)
        draw.text((80, 140), "NON-FAUCET ITEM", fill=(255, 255, 255))

    # Apply variation if query test image (e.g., slight crop or shift)
    if variation == "shifted":
        img = img.crop((10, 10, 290, 290)).resize((width, height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Finishes color representations
    CHROME = (200, 205, 215)
    MATTE_BLACK = (45, 45, 48)
    CHAMPAGNE_BRONZE = (195, 155, 105)
    STAINLESS = (160, 165, 170)

    # Define 6 synthetic catalog models
    models = [
        {
            "model_id": "TEST-MODEL-001",
            "model_number": "TEST-9178-DST",
            "product_title": "Synthetic Leland Pull-Down Kitchen Faucet (Test Fixture)",
            "product_url": "https://example.com/test-fixtures/9178-dst",
            "category": "Kitchen",
            "finish": "Chrome",
            "handle_configuration": "Single Handle",
            "mounting_type": "Deck Mount",
            "spout_type": "High Arc Pull-Down",
            "type": "gooseneck",
            "color": CHROME,
            "has_second_image": True,
        },
        {
            "model_id": "TEST-MODEL-002",
            "model_number": "TEST-9159-BL-DST",
            "product_title": "Synthetic Trinsic Single Handle Pull-Down Faucet (Test Fixture)",
            "product_url": "https://example.com/test-fixtures/9159-bl-dst",
            "category": "Kitchen",
            "finish": "Matte Black",
            "handle_configuration": "Single Handle",
            "mounting_type": "Deck Mount",
            "spout_type": "High Arc Pull-Down",
            "type": "gooseneck",
            "color": MATTE_BLACK,
            "has_second_image": True,
        },
        {
            "model_id": "TEST-MODEL-003",
            "model_number": "TEST-4153-SD-DST",
            "product_title": "Synthetic Linden Pull-Out Kitchen Faucet (Test Fixture)",
            "product_url": "https://example.com/test-fixtures/4153-sd-dst",
            "category": "Kitchen",
            "finish": "Chrome",
            "handle_configuration": "Single Handle",
            "mounting_type": "Deck Mount",
            "spout_type": "Low Arc Pull-Out",
            "type": "pull_out",
            "color": CHROME,
            "has_second_image": False,
        },
        {
            "model_id": "TEST-MODEL-004",
            "model_number": "TEST-3538-CZMPU-DST",
            "product_title": "Synthetic Lahara Two Handle Widespread Bath Faucet (Test Fixture)",
            "product_url": "https://example.com/test-fixtures/3538-czmpu-dst",
            "category": "Bath",
            "finish": "Champagne Bronze",
            "handle_configuration": "Two Handle Widespread",
            "mounting_type": "Deck Mount",
            "spout_type": "Mid Arc",
            "type": "widespread_bath",
            "color": CHAMPAGNE_BRONZE,
            "has_second_image": True,
        },
        {
            "model_id": "TEST-MODEL-005",
            "model_number": "TEST-559LF-SS",
            "product_title": "Synthetic Modern Single Hole Bathroom Faucet (Test Fixture)",
            "product_url": "https://example.com/test-fixtures/559lf-ss",
            "category": "Bath",
            "finish": "Stainless",
            "handle_configuration": "Single Handle",
            "mounting_type": "Single Hole",
            "spout_type": "Contemporary Angular",
            "type": "modern_single_hole",
            "color": STAINLESS,
            "has_second_image": False,
        },
        {
            "model_id": "TEST-MODEL-006",
            "model_number": "TEST-101-DST",
            "product_title": "Synthetic Classic Bridge Two Handle Kitchen Faucet (Test Fixture)",
            "product_url": "https://example.com/test-fixtures/101-dst",
            "category": "Kitchen",
            "finish": "Chrome",
            "handle_configuration": "Two Handle Bridge",
            "mounting_type": "Deck Mount",
            "spout_type": "High Arc",
            "type": "bridge_kitchen",
            "color": CHROME,
            "has_second_image": False,
        },
    ]

    catalog_rows = []
    manifest_rows = []

    for m in models:
        m_id = m["model_id"]
        # Generate primary catalog image
        prim_img_path = IMAGES_DIR / f"{m_id}_primary.png"
        draw_synthetic_faucet(m["type"], m["color"], prim_img_path)

        rel_prim_path = str(prim_img_path.relative_to(BASE_DIR)).replace("\\", "/")

        catalog_rows.append({
            "model_id": m_id,
            "model_number": m["model_number"],
            "product_title": m["product_title"],
            "product_url": m["product_url"],
            "local_path": rel_prim_path,
            "category": m["category"],
            "finish": m["finish"],
            "handle_configuration": m["handle_configuration"],
            "mounting_type": m["mounting_type"],
            "spout_type": m["spout_type"],
            "specifications_json": json.dumps({"test_fixture": True, "canonical_id": m_id}),
            "inventory_status": "Synthetic Demo Feed (In Stock)",
            "source_name": "Synthetic Fixtures Generator",
            "source_checked_at": "2026-10-17T00:00:00Z",
        })

        manifest_rows.append({
            "image_id": f"img_{m_id}_1",
            "model_id": m_id,
            "local_path": rel_prim_path,
            "view_type": "primary",
            "content_hash": "synth_hash_1",
            "download_status": "local_generated",
        })

        # Secondary angle image if flagged
        if m["has_second_image"]:
            sec_img_path = IMAGES_DIR / f"{m_id}_angle.png"
            draw_synthetic_faucet(m["type"], m["color"], sec_img_path, bg_color=(235, 235, 240))
            rel_sec_path = str(sec_img_path.relative_to(BASE_DIR)).replace("\\", "/")
            manifest_rows.append({
                "image_id": f"img_{m_id}_2",
                "model_id": m_id,
                "local_path": rel_sec_path,
                "view_type": "angle",
                "content_hash": "synth_hash_2",
                "download_status": "local_generated",
            })

    # Write products.csv and image_manifest.csv to fixtures and data/
    import pandas as pd
    prod_df = pd.DataFrame(catalog_rows)
    man_df = pd.DataFrame(manifest_rows)

    prod_df.to_csv(FIXTURES_DIR / "products.csv", index=False)
    man_df.to_csv(FIXTURES_DIR / "image_manifest.csv", index=False)

    # Also copy to root data/
    prod_df.to_csv(BASE_DIR / "data" / "products.csv", index=False)
    man_df.to_csv(BASE_DIR / "data" / "image_manifest.csv", index=False)

    # Generate evaluation queries
    eval_queries = []
    # 1. Query for TEST-MODEL-001 (Chrome gooseneck)
    q1_path = EVAL_IMAGES_DIR / "query_test_001.png"
    draw_synthetic_faucet("gooseneck", CHROME, q1_path, variation="shifted")
    eval_queries.append({
        "query_image_path": str(q1_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "true_model_id": "TEST-MODEL-001",
        "is_in_catalog": True,
        "image_quality": "good",
        "category": "Kitchen",
        "notes": "Query photo with slight crop",
    })

    # 2. Query for TEST-MODEL-002 (Matte Black gooseneck)
    q2_path = EVAL_IMAGES_DIR / "query_test_002.png"
    draw_synthetic_faucet("gooseneck", MATTE_BLACK, q2_path, bg_color=(220, 220, 220))
    eval_queries.append({
        "query_image_path": str(q2_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "true_model_id": "TEST-MODEL-002",
        "is_in_catalog": True,
        "image_quality": "good",
        "category": "Kitchen",
        "notes": "Query photo with low lighting background",
    })

    # 3. Query for TEST-MODEL-004 (Champagne bronze widespread bath)
    q3_path = EVAL_IMAGES_DIR / "query_test_004.png"
    draw_synthetic_faucet("widespread_bath", CHAMPAGNE_BRONZE, q3_path, variation="shifted")
    eval_queries.append({
        "query_image_path": str(q3_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "true_model_id": "TEST-MODEL-004",
        "is_in_catalog": True,
        "image_quality": "angled",
        "category": "Bath",
        "notes": "Bathroom faucet query",
    })

    # 4. Out-of-catalog negative query to test conservative abstention
    q_neg_path = EVAL_IMAGES_DIR / "query_negative_object.png"
    draw_synthetic_faucet("negative_sample", (0, 0, 0), q_neg_path)
    eval_queries.append({
        "query_image_path": str(q_neg_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "true_model_id": "",
        "is_in_catalog": False,
        "image_quality": "good",
        "category": "Other",
        "notes": "Negative out-of-catalog test image (abstention expected)",
    })

    eval_df = pd.DataFrame(eval_queries)
    eval_df.to_csv(BASE_DIR / "data" / "evaluation" / "queries.csv", index=False)

    print("Synthetic fixtures generated successfully:")
    print(f"   • Products:   {len(prod_df)} models -> data/products.csv")
    print(f"   • Manifest:   {len(man_df)} images -> data/image_manifest.csv")
    print(f"   • Eval tests: {len(eval_df)} queries -> data/evaluation/queries.csv")


if __name__ == "__main__":
    main()

