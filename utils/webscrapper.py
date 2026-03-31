
from pathlib import Path
import sys
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
import re
from page_data import Page


def extract_headings_and_paragraphs(html_content: str) -> dict[str, list[str]]:
    soup = BeautifulSoup(html_content, "lxml")
    container = soup.select_one("div.main__inner-wrapper")
    if container is None:
        return {"h1": [], "headings": [], "paragraphs": []}
    h1_selectors = [
        "h1",
    ]
    other_heading_selectors = [
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    ]
    paragraph_selectors = [
        "p",
    ]
    # Combined for document-order selection while enabling separate groups
    selectors = h1_selectors + other_heading_selectors + paragraph_selectors
    # Normalize helper: collapse whitespace and lowercase for robust matching
    def _normalize(text: str) -> str:
        return " ".join(text.split()).strip().lower()

    # Phrases to exclude from output
    excluded_phrases = {
        _normalize("Are you with PlaceMakers Trade? Click the icon above to check your trade price."),
        _normalize("Check your Trade Price"),
        _normalize("Alternative Products (0)"),
        _normalize("These items are specifically sourced for you, and cannot be returned unless faulty,read more."),
        _normalize("Here's what we'll do for you:"),
        _normalize("These products are ordered directly from the supplier or another branch for you. Our branch team will call you to confirm if we can source the product and let you know delivery time frames and costs (if any)."),
        _normalize("Once you accept this you will be provided a payment link to complete your purchase."),
        _normalize("Why are products marked Direct from Supplier or POA?")
    }

    h1_elements: list[str] = []
    other_heading_elements: list[str] = []
    paragraph_elements: list[str] = []

    # Collect H1 separately
    for tag in container.select(", ".join(h1_selectors)):
        text = tag.get_text(strip=True)
        if text and _normalize(text) not in excluded_phrases:
            h1_elements.append(text)

    # Collect other headings (H2-H6)
    for tag in container.select(", ".join(other_heading_selectors)):
        text = tag.get_text(strip=True)
        if text and _normalize(text) not in excluded_phrases:
            other_heading_elements.append(text)

    # Collect paragraphs
    for tag in container.select(", ".join(paragraph_selectors)):
        text = tag.get_text(strip=True)
        if text and _normalize(text) not in excluded_phrases:
            paragraph_elements.append(text)

    return {
        "h1": h1_elements,
        "headings": other_heading_elements,
        "paragraphs": paragraph_elements,
    }

def _derive_page_identity(url_or_path: str, h1_list: List[str], page_title: Optional[str]) -> tuple[str, str]:
    # page_id from last non-empty path segment; fallback to 'index'
    segment = url_or_path.rstrip("/").split("/")[-1] or "index"
    page_id = segment.lower()
    # Prefer HTML <title>, else H1, else humanized slug
    if page_title and page_title.strip():
        page_name = page_title.strip()
    elif h1_list:
        page_name = h1_list[0]
    else:
        page_name = page_id.replace("-", " ").title()
    return page_id, page_name


def extract_content(html_source: str) -> Page:
    try:
        if html_source.startswith("http://") or html_source.startswith("https://"):
            response = requests.get(html_source, timeout=30)
            response.raise_for_status()
            # Ensure text is decoded properly
            if not response.encoding:
                response.encoding = response.apparent_encoding
            html_content = response.text
        else:
            path = Path(html_source)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            html_content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"Error fetching or reading HTML: {exc}")
        return None

    results = extract_headings_and_paragraphs(html_content)
    h1_list = results.get("h1", [])
    headings_list = results.get("headings", [])
    paragraphs_list = results.get("paragraphs", [])

    # Extract <title> from HTML for page name if available
    soup_for_title = BeautifulSoup(html_content, "lxml")
    page_title_tag = soup_for_title.find("title")
    page_title_text = page_title_tag.get_text(strip=True) if page_title_tag else None

    page_id, page_name = _derive_page_identity(html_source, h1_list, page_title_text)

    # Extract products from carousel component
    products_list: list[dict] = []
    carousel = soup_for_title.select_one("div.carousel__component")
    if carousel is not None:
        for item in carousel.select("div.carousel__item"):
            sku = item.select_one(".carousel__item--code")
            name = item.select_one(".carousel__item--name")
            brand = item.select_one(".carousel__item-manufacturer")
            # Remove nested div.partCode and strip 'SKU:' label (case-insensitive)
            if sku:
                nested_part = sku.select_one("div.partCode")
                if nested_part is not None:
                    nested_part.extract()
                sku_text = sku.get_text(strip=True)
                sku_text = re.sub(r"(?i)^sku\s*:\s*", "", sku_text).strip()
            else:
                sku_text = ""
            name_text = name.get_text(strip=True) if name else ""
            brand_text = brand.get_text(strip=True) if brand else ""
            if any([sku_text, name_text, brand_text]):
                products_list.append({
                    "sku": sku_text,
                    "name": name_text,
                    "brand": brand_text,
                })

    return Page(
        page_id=page_id,
        page_name=page_name,
        heading=h1_list[0] if h1_list else "",
        sub_headings=headings_list,
        paragraphs=paragraphs_list,
        products=products_list,
        tags=[],
        projects=[],
        url=html_source,
    )