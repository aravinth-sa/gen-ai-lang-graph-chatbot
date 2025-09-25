from pathlib import Path
from typing import List
import time

from websource import get_page_urls
from webscrapper import extract_content
from data_transformation import to_json
from page_data import Page


def prepare_data() -> str:
    # Read URLs from content_source.txt file
    content_source_path = Path("dataset/input/content_source.txt")
    contents = []
    
    with open(content_source_path, 'r') as file:
        for line in file:
            url = line.strip()
            if url:  # Skip empty lines
                # Add leading slash if not present
                if not url.startswith('/'):
                    url = '/' + url
                contents.append(url)

    #contents=['/trade-deals']
    xml_path = str(Path("dataset/input/content-pages.xml"))
    matching_urls = get_page_urls(xml_path, contents)

    pages: List[Page] = []
    batch_size = 2  # Process 2 URLs per batch
    
    for i in range(0, len(matching_urls), batch_size):
        batch_urls = matching_urls[i:i+batch_size]
        
        # Process each URL in the current batch
        for url in batch_urls:
            try:
                print(f"Extracting content from: {url}")
                page = extract_content(url)
                pages.append(page)
            except Exception as e:
                print(f"Error extracting content from {url}: {str(e)}")
                print(f"Skipping URL and continuing with the next one...")
                continue
        
        # Wait for 2 seconds before processing the next batch
        # Skip waiting after the last batch
        if i + batch_size < len(matching_urls):
            print(f"Processed {i+len(batch_urls)} of {len(matching_urls)} URLs. Waiting 2 seconds...")
            time.sleep(2)

    output_json = str(Path("dataset/output/output-raw-page_v2.json"))
    to_json(pages, output_json)
    return output_json

if __name__ == "__main__":
    prepare_data()

