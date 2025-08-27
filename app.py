import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def extract_headings_and_paragraphs(html_content: str) -> list[str]:
    soup = BeautifulSoup(html_content, "lxml")
    container = soup.select_one("div.main__inner-wrapper")
    if container is None:
        return []
    selectors = [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
    ]
    # Normalize helper: collapse whitespace and lowercase for robust matching
    def _normalize(text: str) -> str:
        return " ".join(text.split()).strip().lower()

    # Phrases to exclude from output
    excluded_phrases = {
        _normalize("Are you with PlaceMakers Trade? Click the icon above to check your trade price."),
        _normalize("Check your Trade Price"),
        _normalize("Alternative Products (0)"),
    }

    elements: list[str] = []
    # Collect headings and paragraphs INSIDE the container, in document order
    for tag in container.select(", ".join(selectors)):
        text = tag.get_text(strip=True)
        if text and _normalize(text) not in excluded_phrases:
            elements.append(text)

    return elements


def main() -> None:
    html_source = ""

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
        sys.exit(3)

    items = extract_headings_and_paragraphs(html_content)
    for item in items:
        print(item)


if __name__ == "__main__":
    main()


