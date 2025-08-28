from pathlib import Path
from typing import List

from utils.websource import get_page_urls
from utils.webscrapper import extract_content
from utils.data_transformation import to_json
from utils.page_data import Page


def prepare_data() -> str:
    contents = [
        "/projects/kitchens",
        "/projects/bathrooms",
        "/projects/heating-and-cooling",
        "/projects/landscaping",
        "/projects/cladding",
        "/projects/insulation",
        "/projects/plywood",
    ]

    xml_path = str(Path("dataset/input/content-pages.xml"))
    matching_urls = get_page_urls(xml_path, contents)

    pages: List[Page] = []
    for url in matching_urls:
        page = extract_content(url)
        pages.append(page)

    output_json = str(Path("dataset/output/output-raw-page.json"))
    to_json(pages, output_json)
    return output_json


