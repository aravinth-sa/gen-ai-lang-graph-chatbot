from typing import List
import csv
import json
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


def to_json(pages: List[Page], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for p in pages:
        payload.append(
            {
                "page_id": p.page_id,
                "page_name": p.page_name,
                "heading": p.heading,
                "sub_headings": p.sub_headings or [],
                "paragraphs": p.paragraphs or [],
                "products": p.products or [],
                "tags": p.tags or [],
                "projects": p.projects or [],
                "url": p.url,
            }
        )
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

