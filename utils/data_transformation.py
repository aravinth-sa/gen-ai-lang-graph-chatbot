from typing import List
import csv
from pathlib import Path
from utils.page_data import Page


def to_csv(pages: List[Page], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "page_id",
        "page_name",
        "heading",
        "sub_headings",
        "paragraphs",
        "products",
        "tags",
        "projects",
        "url",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in pages:
            writer.writerow(
                {
                    "page_id": p.page_id,
                    "page_name": p.page_name,
                    "heading": p.heading,
                    # Join lists with pipe for readability; safe for CSV
                    "sub_headings": " | ".join(p.sub_headings or []),
                    "paragraphs": " | ".join(p.paragraphs or []),
                    "products": " | ".join(p.products or []),
                    "tags": " | ".join(p.tags or []),
                    "projects": " | ".join(p.projects or []),
                    "url": p.url,
                }
            )

