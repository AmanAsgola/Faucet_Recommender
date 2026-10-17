"""Inspect catalogue pages for layout of text blocks and images."""

import sys
from pathlib import Path
import pymupdf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pdf_path = Path("Kitchen & Bath Full-Line Catalogue.pdf")
if not pdf_path.is_file():
    print("PDF not found:", pdf_path)
    sys.exit(1)

doc = pymupdf.open(str(pdf_path))
print(f"Total Pages: {len(doc)}")

# Check pages 12, 13, 15 (sample kitchen pages)
sample_pages = [11, 12, 14, 25, 27]

for pno in sample_pages:
    page = doc[pno]
    print(f"\n=================== PAGE {pno + 1} ===================")
    blocks = page.get_text("blocks")
    print(f"Text blocks count: {len(blocks)}")
    for b in blocks[:8]:
        text = b[4].strip().replace("\n", " ")
        if text:
            print(f"  Block ({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}): {text[:100]}")

    image_list = page.get_images()
    print(f"\nImages on page: {len(image_list)}")
    for img_idx, img_info in enumerate(image_list[:8]):
        xref = img_info[0]
        rects = page.get_image_rects(xref)
        bi = doc.extract_image(xref)
        w, h, ext = bi["width"], bi["height"], bi["ext"]
        rect_str = f"({rects[0].x0:.1f}, {rects[0].y0:.1f}, {rects[0].x1:.1f}, {rects[0].y1:.1f})" if rects else "None"
        print(f"  Img #{img_idx} (xref={xref}): {w}x{h} {ext}, Pos={rect_str}")

