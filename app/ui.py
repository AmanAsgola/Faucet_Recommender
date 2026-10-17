"""Streamlit UI for Faucet Finder POC."""

from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from app.config import settings
from app.service import FaucetIdentificationService

# Page configuration
st.set_page_config(
    page_title="Faucet Finder — POC",
    page_icon="🚿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean presentation
st.markdown(
    """
    <style>
    .candidate-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        background-color: #fafafa;
    }
    .badge-likely {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
        display: inline-block;
    }
    .badge-confirm {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
        display: inline-block;
    }
    .badge-nomatch {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        margin: 16px 0;
        border-radius: 4px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing Faucet Identification Service...")
def get_service() -> FaucetIdentificationService:
    """Initialize and cache service instance."""
    return FaucetIdentificationService()


def render_uncertainty_badge(label: str) -> str:
    """Return HTML badge for uncertainty label."""
    if label == settings.label_likely_match:
        return f'<span class="badge-likely">✓ {label.replace("_", " ").upper()}</span>'
    elif label == settings.label_needs_confirmation:
        return f'<span class="badge-confirm">⚠ {label.replace("_", " ").upper()}</span>'
    else:
        return f'<span class="badge-nomatch">✕ {label.replace("_", " ").upper()}</span>'


def main():
    st.title("🚿 Faucet Finder — POC")
    st.caption("Visual Faucet Identification Proof of Concept for Authorized Catalog Data")

    # Important notice
    st.markdown(
        f'<div class="disclaimer-box">⚠️ <strong>Notice:</strong> {settings.repair_part_disclaimer}</div>',
        unsafe_allow_html=True,
    )

    # Initialize service
    service = get_service()
    system_ready, ready_msg = service.check_system_ready()

    # Sidebar: Index Status & Controls
    with st.sidebar:
        st.header("⚙️ System Status")
        if system_ready:
            st.success("Catalog & FAISS Index: Ready")
            if service.retriever.manifest:
                st.caption(f"**Indexed Vectors:** {service.retriever.manifest.get('vector_count', 'N/A')}")
                st.caption(f"**Model:** {service.retriever.manifest.get('model_name', 'OpenCLIP')}")
                st.caption(f"**Built:** {service.retriever.manifest.get('build_timestamp', 'N/A')[:19]}")
        else:
            st.error(ready_msg)
            st.info(
                "To build the catalog index, run:\n"
                "```bash\n"
                "python scripts/validate_catalog.py\n"
                "python scripts/build_index.py\n"
                "```"
            )

        st.divider()
        st.header("🔍 Search Settings")
        use_hybrid = st.toggle("Enable Hybrid Reranking", value=True, help="Combine visual similarity with text and metadata attributes")
        
        st.subheader("Optional Context")
        query_text = st.text_input(
            "Keywords / Description",
            placeholder="e.g. Single handle pull-down chrome",
            help="Optional text to enhance visual matching",
        )
        cat_filter = st.selectbox(
            "Category",
            options=["", "Kitchen", "Bath", "Bar/Prep", "Laundry"],
            help="Filter/match category if known",
        )
        finish_filter = st.selectbox(
            "Finish",
            options=["", "Chrome", "Matte Black", "Stainless", "Champagne Bronze", "Brushed Nickel"],
            help="Filter/match finish if known",
        )

        st.divider()
        st.caption(
            "**Data Compliance:** No scraping or automated crawling of Delta Faucet website. "
            "Data strictly loaded from local authorized catalog feeds."
        )

    # Main search area
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        st.subheader("1. Upload Faucet Photo")
        uploaded_file = st.file_uploader(
            "Choose a faucet photo (JPEG, PNG, or WebP, max 10MB)",
            type=["jpg", "jpeg", "png", "webp"],
            help="For best results, capture a clear photo of the faucet spout, handles, and base.",
        )

    preview_image = None
    with col_preview:
        st.subheader("2. Photo Preview")
        if uploaded_file is not None:
            try:
                preview_image = Image.open(uploaded_file)
                st.image(preview_image, caption="Uploaded Photo", use_container_width=True)
            except Exception as e:
                st.error(f"Cannot display uploaded image: {e}")
        else:
            st.info("Upload a photo above to see preview.")

    # Search trigger button
    search_clicked = st.button("🔎 Find Matching Models", type="primary", disabled=(uploaded_file is None or not system_ready))

    if search_clicked and uploaded_file is not None:
        attributes = {}
        if cat_filter:
            attributes["category"] = cat_filter
        if finish_filter:
            attributes["finish"] = finish_filter

        with st.spinner("Analyzing faucet features and searching catalog..."):
            result = service.identify(
                image_input=uploaded_file,
                query_text=query_text if query_text else None,
                query_attributes=attributes if attributes else None,
                use_hybrid_reranking=use_hybrid,
                top_k=settings.display_top_k,
            )

        st.divider()

        # Display result status and guidance
        if result.status == "error":
            st.error(f"**Error:** {result.error_message or result.guidance_message}")
            return

        st.subheader("3. Identification Results")

        # Guidance alert
        if result.overall_confidence == settings.label_likely_match:
            st.success(f"**Match Assessment:** {result.guidance_message}")
        elif result.overall_confidence == settings.label_needs_confirmation:
            st.warning(f"**Match Assessment:** {result.guidance_message}")
        else:
            st.error(f"**Match Assessment:** {result.guidance_message}")

        if not result.candidates:
            st.info(
                "No candidate models returned. Please try another photo from a different angle "
                "or ensure the catalog index contains valid items."
            )
            return

        st.write(f"Showing top {len(result.candidates)} candidates:")

        # Show candidate cards
        cols = st.columns(len(result.candidates))

        for idx, (col, cand) in enumerate(zip(cols, result.candidates), start=1):
            with col:
                st.markdown(
                    f"### #{idx}: {cand['model_number']}\n"
                    f"{render_uncertainty_badge(cand['uncertainty_label'])}",
                    unsafe_allow_html=True,
                )
                
                # Product Image Thumbnail
                img_path = cand.get("best_image_path")
                img_displayed = False
                if img_path:
                    p = Path(img_path)
                    if not p.is_absolute():
                        p = settings.base_dir / p
                    if p.is_file():
                        st.image(str(p), caption=f"Catalog Image ({cand['model_number']})", use_container_width=True)
                        img_displayed = True

                if not img_displayed and cand.get("image_url"):
                    st.image(cand["image_url"], caption=f"Catalog Image ({cand['model_number']})", use_container_width=True)
                    img_displayed = True

                if not img_displayed:
                    st.caption("*(No image file available in catalog)*")

                st.markdown(f"**{cand['product_title']}**")

                # Attributes
                attrs_text = []
                if cand.get("category"):
                    attrs_text.append(f"**Category:** {cand['category']}")
                if cand.get("finish"):
                    attrs_text.append(f"**Finish:** {cand['finish']}")
                if cand.get("handle_configuration"):
                    attrs_text.append(f"**Handles:** {cand['handle_configuration']}")
                if cand.get("mounting_type"):
                    attrs_text.append(f"**Mount:** {cand['mounting_type']}")
                if cand.get("spout_type"):
                    attrs_text.append(f"**Spout:** {cand['spout_type']}")

                if attrs_text:
                    st.markdown("\n".join(attrs_text))

                # Inventory status (Authoritative only)
                st.caption(f"**Inventory:** {cand.get('inventory_status')}")

                # Score & Evidence
                st.metric(
                    label="Composite Similarity Score",
                    value=f"{cand['final_score']:.3f}",
                    help="Similarity metric (0.000 to 1.000). Note: this is a geometric vector similarity, not an accuracy probability.",
                )

                with st.expander("Evidence & Signal Breakdown"):
                    for ev in cand.get("evidence", []):
                        st.write(f"- {ev}")
                    st.caption(f"Active Weights: {cand.get('active_weights')}")

                # Product URL Link
                if cand.get("product_url"):
                    st.link_button("🌐 View Product Page", cand["product_url"], use_container_width=True)

                st.divider()

                # User Feedback Buttons
                st.write("**Was this candidate accurate?**")
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    if st.button("👍 Correct", key=f"fb_pos_{cand['model_id']}"):
                        st.toast(f"Thank you! Recorded positive feedback for {cand['model_number']}.")
                with fb_col2:
                    if st.button("👎 Wrong", key=f"fb_neg_{cand['model_id']}"):
                        st.toast(f"Recorded negative feedback for {cand['model_number']}.")


if __name__ == "__main__":
    main()

