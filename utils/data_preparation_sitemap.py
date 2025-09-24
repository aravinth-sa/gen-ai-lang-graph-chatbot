from pathlib import Path
from typing import List

from websource import get_page_urls
from webscrapper import extract_content
from data_transformation import to_json
from page_data import Page


def prepare_data() -> str:
    contents = [
        "/projects/kitchens",
        "/projects/bathrooms",
        "/projects/heating-and-cooling",
        "/projects/landscaping",
        "/projects/cladding",
        "/projects/insulation",
        "/projects/plywood",
        "/timber-1",
        "/projects/fastenings",
        "/interior-wall-ceiling-linings",
        "/projects/building-envelope-barriers",
        "/projects/laundry",
        "/wardrobes",
        "/paint",
        "/farm-sheds",
        "/projects/rural-supplies"
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

if __name__ == "__main__":
    prepare_data()

