"""
Extract product web features from HTML table and convert to JSON format.

This script scrapes data from the queryResultTable in webfeatue-result.html,
removes the prefix "placemakersClassification/1.0/WebFeature." from qualifiers,
and outputs a JSON file with products and their features.
"""

import json
from bs4 import BeautifulSoup
from collections import defaultdict
import os


def extract_webfeature_data(html_file_path, output_json_path):
    """
    Extract web feature data from HTML table and save as JSON.
    
    Args:
        html_file_path: Path to the input HTML file
        output_json_path: Path to the output JSON file
    """
    print(f"Reading HTML file: {html_file_path}")
    
    # Read the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the table with id="queryResultTable"
    table = soup.find('table', {'id': 'queryResultTable'})
    
    if not table:
        raise ValueError("Table with id 'queryResultTable' not found in HTML file")
    
    # Extract table rows
    tbody = table.find('tbody')
    if not tbody:
        raise ValueError("Table body not found")
    
    rows = tbody.find_all('tr')
    print(f"Found {len(rows)} rows in the table")
    
    # Dictionary to store products and their features
    products_dict = defaultdict(dict)
    
    # Process each row
    prefix_to_remove = "placemakersClassification/1.0/WebFeature."
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 3:
            code = cells[0].get_text(strip=True)
            qualifier = cells[1].get_text(strip=True)
            value = cells[2].get_text(strip=True)
            
            # Remove the prefix from qualifier
            if qualifier.startswith(prefix_to_remove):
                feature_name = qualifier.replace(prefix_to_remove, '')
            else:
                feature_name = qualifier
            
            # Add to products dictionary
            if 'code' not in products_dict[code]:
                products_dict[code]['code'] = code
            
            products_dict[code][feature_name] = value
    
    # Convert to list of products
    products_list = list(products_dict.values())
    
    print(f"Extracted {len(products_list)} unique products")
    
    # Save to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(products_list, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved data to: {output_json_path}")
    
    # Print sample of first product
    if products_list:
        print("\nSample product (first item):")
        print(json.dumps(products_list[0], indent=2))
    
    return products_list


if __name__ == "__main__":
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    html_file = os.path.join(project_root, 'dataset', 'input', 'webfeatue-result.html')
    output_file = os.path.join(project_root, 'dataset', 'output', 'product_v2.json')
    
    # Extract data
    extract_webfeature_data(html_file, output_file)
