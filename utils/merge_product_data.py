import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def load_product_metadata_v2(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load product_metadata_v2.json and index by code."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Index products by code
    products_by_code = {}
    for product in data.get('products', []):
        code = product.get('code')
        if code:
            products_by_code[code] = product
    
    print(f"Loaded {len(products_by_code)} products from product_metadata_v2.json")
    return products_by_code


def load_product_v2(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load product_v2.json and index by code."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Index products by code
    products_by_code = {}
    for product in data:
        code = product.get('code')
        if code:
            products_by_code[code] = product
    
    print(f"Loaded {len(products_by_code)} products from product_v2.json")
    return products_by_code


def merge_products(metadata_products: Dict[str, Dict[str, Any]], 
                   v2_products: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge products from both sources based on code."""
    merged_products = []
    
    # Get all unique codes from both sources
    all_codes = set(metadata_products.keys()) | set(v2_products.keys())
    
    for code in sorted(all_codes):
        merged_product = {'code': code}
        
        # Get data from both sources
        metadata = metadata_products.get(code, {})
        v2_data = v2_products.get(code, {})
        
        # Merge metadata fields (from product_metadata_v2.json)
        if metadata:
            merged_product.update({
                'title': metadata.get('title', ''),
                'description': metadata.get('description', ''),
                'thumb_image': metadata.get('thumb_image', ''),
                'url': metadata.get('url', ''),
                'keywords': '|'.join(metadata.get('keywords', [])) if isinstance(metadata.get('keywords'), list) else metadata.get('keywords', ''),
                'category': '|'.join(metadata.get('category', [])) if isinstance(metadata.get('category'), list) else metadata.get('category', ''),
                'brand': metadata.get('brand', ''),
                'subBrand': metadata.get('subBrand', ''),
                'subClassName': metadata.get('subClassName', ''),
                'discountPrice': metadata.get('discountPrice', '')
            })
        
        # Merge v2 fields (from product_v2.json)
        if v2_data:
            # Add all fields from v2_data except 'code'
            for key, value in v2_data.items():
                if key != 'code':
                    merged_product[key] = value
        
        merged_products.append(merged_product)
    
    print(f"Merged {len(merged_products)} unique products")
    return merged_products


def write_to_csv(products: List[Dict[str, Any]], output_file: Path):
    """Write merged products to CSV file."""
    if not products:
        print("No products to write")
        return
    
    # Get all unique field names across all products
    fieldnames = set()
    for product in products:
        fieldnames.update(product.keys())
    
    # Sort fieldnames with 'code' first
    fieldnames = ['code'] + sorted([f for f in fieldnames if f != 'code'])
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(products)
    
    print(f"Successfully wrote {len(products)} products to {output_file}")


def main():
    """Main function to merge product data and create CSV."""
    # Define paths
    current_dir = Path(__file__).parent
    dataset_dir = current_dir.parent / 'dataset' / 'output'
    
    metadata_file = dataset_dir / 'product_metadata_v2.json'
    v2_file = dataset_dir / 'product_v2.json'
    output_file = dataset_dir / 'product_v2.csv'
    
    # Check if input files exist
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found")
        return
    
    if not v2_file.exists():
        print(f"Error: {v2_file} not found")
        return
    
    # Load data from both files
    print("Loading product data...")
    metadata_products = load_product_metadata_v2(metadata_file)
    v2_products = load_product_v2(v2_file)
    
    # Compare codes
    metadata_codes = set(metadata_products.keys())
    v2_codes = set(v2_products.keys())
    
    print(f"\nCode comparison:")
    print(f"  Codes in metadata only: {len(metadata_codes - v2_codes)}")
    print(f"  Codes in v2 only: {len(v2_codes - metadata_codes)}")
    print(f"  Codes in both: {len(metadata_codes & v2_codes)}")
    print(f"  Total unique codes: {len(metadata_codes | v2_codes)}")
    
    # Merge products
    print("\nMerging products...")
    merged_products = merge_products(metadata_products, v2_products)
    
    # Write to CSV
    print(f"\nWriting to CSV...")
    write_to_csv(merged_products, output_file)
    
    print(f"\nComplete! Final CSV saved to: {output_file}")


if __name__ == '__main__':
    main()
