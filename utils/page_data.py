from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Page:
    page_id: str
    page_name: str
    heading: str
    sub_headings: List[str]
    paragraphs: List[str]
    products: List[dict]
    tags: List[str]
    projects: List[dict]
    url: str
