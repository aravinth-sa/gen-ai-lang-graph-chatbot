import json
import os
from pathlib import Path

def extract_products():
    """Extract product data from JSONL file and transform according to requirements.
    
    1. Map path to code (excluding '/products/')
    2. Map title, description, thumb_image, url, keywords directly
    3. Map category_paths to category (excluding root value)
    """
    # Define input and output paths
    current_dir = Path(__file__).parent
    input_file = current_dir.parent / 'dataset' / 'input' / 'ConsumerProductCatalog_20250915.jsonl'
    output_dir = current_dir.parent / 'dataset' / 'output'
    output_file = output_dir / 'product.json'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize products list
    products = []
    
    # Read and process JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Parse JSON line
                product_data = json.loads(line)
                
                # Check if this is a product entry
                if product_data.get('op') == 'add' and '/products/' in product_data.get('path', ''):
                    # Extract product value
                    product_value = product_data.get('value', {})
                    attributes = product_value.get('attributes', {})
                    
                    # Extract path and convert to code (removing '/products/')
                    path = product_data.get('path', '')
                    code = path.replace('/products/', '')
                    
                    # Extract category paths (excluding root)
                    categories = []
                    category_paths = attributes.get('category_paths', [])
                    for path_list in category_paths:
                        # Skip empty paths
                        if not path_list:
                            continue
                        
                        # Filter out 'root' entries
                        filtered_categories = [cat for cat in path_list if cat.get('name') != 'root']
                        if filtered_categories:
                            categories.extend([cat.get('name') for cat in filtered_categories])
                    
                    # Create transformed product
                    transformed_product = {
                        'code': code,
                        'title': attributes.get('title', ''),
                        'description': attributes.get('description', ''),
                        'thumb_image': attributes.get('thumb_image', ''),
                        'url': attributes.get('url', ''),
                        'keywords': attributes.get('keywords', []),
                        'category': categories
                    }
                    
                    # Add to products list
                    products.append(transformed_product)
            except json.JSONDecodeError:
                print(f"Error parsing JSON line: {line[:100]}...")
                continue
    
    # Write products to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'products': products}, f, indent=2)
    
    print(f"Extracted {len(products)} products to {output_file}")
    return products

if __name__ == '__main__':
    extract_products()
