from pathlib import Path
from typing import List

from data.websource import get_matching_loc_values
from data.webscrapper import extract_content
from data.data_transformation import to_csv
from data.page_data import Page


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

    xml_path = str(Path("dataset/content-pages.xml"))
    matching_urls = get_matching_loc_values(xml_path, contents)

    pages: List[Page] = []
    for url in matching_urls:
        page = extract_content(url)
        pages.append(page)

    output_csv = str(Path("dataset/output/pages.csv"))
    to_csv(pages, output_csv)
    return output_csv


