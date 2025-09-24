#!/usr/bin/env python3
"""
JSON to CSV Converter

This script converts large JSON files to CSV format, with special handling for the product.json structure.
It processes the file in chunks to avoid memory issues with large files.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Iterator


def read_json_in_chunks(file_path: str, chunk_size: int = 1000) -> Iterator[List[Dict]]:
    """
    Read a large JSON file in chunks to avoid memory issues.
    
    Args:
        file_path: Path to the JSON file
        chunk_size: Number of items to process at once
        
    Yields:
        Dictionary objects from the JSON file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try to load the entire file as JSON first
        try:
            data = json.load(f)
            if 'products' in data:
                # Process products in chunks
                products = data['products']
                for i in range(0, len(products), chunk_size):
                    yield products[i:i + chunk_size]
                return
        except (json.JSONDecodeError, MemoryError):
            # If the file is too large or malformed, fall back to manual parsing
            f.seek(0)
            
        # Manual parsing for large files
        # Read the opening bracket
        char = f.read(1)
        if char != '{':
            raise ValueError("Expected JSON object to start with '{'")
        
        # Find the products array
        while True:
            chunk = f.read(1000)
            if not chunk:
                return
            products_pos = chunk.find('"products"')
            if products_pos >= 0:
                # Position file pointer after "products": [
                f.seek(f.tell() - len(chunk) + products_pos)
                break
        
        # Skip until we find the opening bracket of the array
        while True:
            char = f.read(1)
            if not char:
                return
            if char == '[':
                break
        
        # Now we're at the start of the products array
        buffer = ""
        depth = 0
        item_count = 0
        items = []
        
        while True:
            char = f.read(1)
            if not char:  # End of file
                break
                
            buffer += char
            
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:  # End of an object
                    # Process the item (remove trailing comma if present)
                    item_json = buffer.rstrip(',')
                    try:
                        # Fix common JSON parsing issues
                        # 1. Make sure the item is a complete JSON object
                        if not item_json.strip().startswith('{'):
                            item_json = '{' + item_json
                        if not item_json.strip().endswith('}'):
                            item_json = item_json + '}'
                            
                        # 2. Handle truncated strings by ensuring quotes are properly closed
                        quote_count = item_json.count('"')
                        if quote_count % 2 != 0:
                            item_json = item_json + '"'
                            
                        item = json.loads(item_json)
                        items.append(item)
                        item_count += 1
                    except json.JSONDecodeError:
                        # Try a more aggressive approach to fix the JSON
                        try:
                            # Extract key fields using regex
                            import re
                            code_match = re.search(r'"code"\s*:\s*"([^"]+)"', item_json)
                            title_match = re.search(r'"title"\s*:\s*"([^"]+)"', item_json)
                            
                            if code_match and title_match:
                                # Create a minimal valid item with just code and title
                                item = {
                                    'code': code_match.group(1),
                                    'title': title_match.group(1)
                                }
                                items.append(item)
                                item_count += 1
                            else:
                                print(f"Error parsing JSON item: {item_json[:100]}...")
                        except Exception:
                            print(f"Error parsing JSON item: {item_json[:100]}...")
                    
                    buffer = ""
                    
                    # Yield batch if we've reached chunk_size
                    if item_count >= chunk_size:
                        yield items
                        items = []
                        item_count = 0
            
            # Check if we've reached the end of the array
            if depth == 0 and char == ']':
                break
        
        # Yield any remaining items
        if items:
            yield items


def json_to_csv(json_file: str, csv_file: str, flatten_lists: bool = True) -> None:
    """
    Convert a JSON file to CSV format.
    
    Args:
        json_file: Path to the input JSON file
        csv_file: Path to the output CSV file
        flatten_lists: Whether to flatten list fields into pipe-separated strings
    """
    print(f"Converting {json_file} to {csv_file}...")
    
    # Create output directory if it doesn't exist
    output_path = Path(csv_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the first chunk to determine headers
    first_chunk = next(read_json_in_chunks(json_file, 1))
    if not first_chunk:
        print("Error: No data found in JSON file")
        return
    
    # Extract all possible fields from the first chunk
    all_fields = set()
    for item in first_chunk:
        all_fields.update(item.keys())
    
    fieldnames = sorted(list(all_fields))
    print(f"Fields detected: {', '.join(fieldnames)}")
    
    # Write CSV header and first chunk
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process the first chunk
        for item in first_chunk:
            row = {}
            for field in fieldnames:
                value = item.get(field, "")
                
                # Handle lists by joining with pipe character if flatten_lists is True
                if flatten_lists and isinstance(value, list):
                    if all(isinstance(x, str) for x in value):
                        row[field] = " | ".join(value)
                    elif all(isinstance(x, dict) for x in value):
                        # For lists of dictionaries, convert to JSON string
                        row[field] = json.dumps(value)
                    else:
                        row[field] = " | ".join(str(x) for x in value)
                else:
                    row[field] = value
            
            writer.writerow(row)
    
    # Process remaining chunks and append to CSV
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Start from the second chunk
        chunk_num = 1
        for chunk in read_json_in_chunks(json_file):
            chunk_num += 1
            print(f"Processing chunk {chunk_num} ({len(chunk)} items)")
            
            for item in chunk:
                row = {}
                for field in fieldnames:
                    value = item.get(field, "")
                    
                    # Handle lists by joining with pipe character if flatten_lists is True
                    if flatten_lists and isinstance(value, list):
                        if all(isinstance(x, str) for x in value):
                            row[field] = " | ".join(value)
                        elif all(isinstance(x, dict) for x in value):
                            # For lists of dictionaries, convert to JSON string
                            row[field] = json.dumps(value)
                        else:
                            row[field] = " | ".join(str(x) for x in value)
                    else:
                        row[field] = value
                
                writer.writerow(row)
    
    print(f"Conversion complete. CSV file saved to {csv_file}")


def convert_product_json(input_file: str, output_file: str) -> None:
    """
    Convert the product.json file to CSV format.
    
    Args:
        input_file: Path to the product.json file
        output_file: Path to the output CSV file
    """
    try:
        # First try a direct approach for smaller files
        with open(input_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if 'products' in data and isinstance(data['products'], list):
                    products = data['products']
                    
                    # Extract all possible fields
                    all_fields = set()
                    for product in products:
                        all_fields.update(product.keys())
                    
                    fieldnames = sorted(list(all_fields))
                    
                    # Write to CSV
                    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for product in products:
                            row = {}
                            for field in fieldnames:
                                value = product.get(field, "")
                                
                                # Handle lists
                                if isinstance(value, list):
                                    if all(isinstance(x, str) for x in value):
                                        row[field] = " | ".join(value)
                                    elif all(isinstance(x, dict) for x in value):
                                        row[field] = json.dumps(value)
                                    else:
                                        row[field] = " | ".join(str(x) for x in value)
                                else:
                                    row[field] = value
                            
                            writer.writerow(row)
                    
                    print(f"Conversion complete. CSV file saved to {output_file}")
                    return
            except (json.JSONDecodeError, MemoryError):
                # Fall back to chunked processing for large files
                pass
    except Exception as e:
        print(f"Error with direct conversion: {str(e)}. Falling back to chunked processing.")
    
    # Fall back to chunked processing
    json_to_csv(input_file, output_file)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JSON file to CSV')
    parser.add_argument('input', help='Input JSON file path')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--no-flatten', action='store_false', dest='flatten',
                        help='Do not flatten list fields into pipe-separated strings')
    
    args = parser.parse_args()
    
    json_to_csv(args.input, args.output, args.flatten)
