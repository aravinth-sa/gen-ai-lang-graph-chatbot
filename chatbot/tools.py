"""
Tools for generating HTML content from product data.
"""

import re
import json
from typing import List, Any
from pathlib import Path

def get_product_image_from_json(product_code: str) -> str:
    """
    Fetches the thumb_image URL from product.json file based on product code.
    
    Args:
        product_code: The product code/SKU to search for
        
    Returns:
        str: The thumb_image URL or empty string if not found
    """
    try:
        # Get the path to product.json
        current_dir = Path(__file__).parent.parent
        product_json_path = current_dir / 'dataset' / 'output' / 'product.json'
        
        if not product_json_path.exists():
            print(f"Product JSON file not found at: {product_json_path}")
            return ''
        
        # Load and search the product.json file
        with open(product_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Search for the product by code
        if 'products' in data:
            for product in data['products']:
                if product.get('code') == product_code:
                    return product.get('thumb_image', '')
        
        print(f"Product code {product_code} not found in product.json")
        return ''
        
    except Exception as e:
        print(f"Error fetching product image from JSON: {e}")
        return ''

def add_sku_hyperlink(text: str) -> str:
    """
    Adds hyperlinks to SKU numbers in the text.
    
    Args:
        text: The input text that may contain SKU numbers in the format "SKU: xxx"
        
    Returns:
        str: The text with SKU numbers converted to Markdown-style links
    """
    # This pattern matches "SKU: " followed by alphanumeric characters
    sku_pattern = r'SKU: ([a-zA-Z0-9]+)'
    
    def replace_sku(match):
        sku = match.group(1)
        return f'[SKU: {sku}](https://www.placemakers.co.nz/online/p/{sku})'
    
    # Replace all occurrences of SKU patterns with Markdown links
    return re.sub(sku_pattern, replace_sku, text)

def format_product_card(product_data: dict) -> str:
    """
    Formats a product as a card with image, title, SKU, and quick view link.
    
    Args:
        product_data: Dictionary containing product information (code, title, thumb_image, url)
        
    Returns:
        str: HTML-formatted product card
    """
    code = product_data.get('code', 'N/A')
    title = product_data.get('title', 'Product')
    thumb_image = product_data.get('thumb_image', '')
    url = product_data.get('url', f'https://www.placemakers.co.nz/online/p/{code}')
    
    # Create HTML product card (inline-block for horizontal layout)
    card_html = f"""
<div style="display: inline-block; vertical-align: top; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin: 10px; max-width: 300px; font-family: Arial, sans-serif;">
    <div style="position: relative;">
        <img src="{thumb_image}" alt="{title}" style="width: 100%; height: auto; border-radius: 4px; background: #f5f5f5;" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22%3E%3Crect fill=%22%23f0f0f0%22 width=%22200%22 height=%22200%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 fill=%22%23999%22%3ENo Image%3C/text%3E%3C/svg%3E';"/>
    </div>
    <h3 style="font-size: 14px; margin: 12px 0 8px 0; color: ##091c5a; line-height: 1.4;">
        <a href="{url}" target="_blank" style="color: ##091c5a; text-decoration: none;">{title}</a>
    </h3>
    <p style="margin: 8px 0; font-size: 13px; color: ##091c5a;">
        <strong>SKU:</strong> <a href="{url}" target="_blank" style="color: #66b3ff; text-decoration: none;">{code}</a>
    </p>
</div>
"""
    return card_html

def format_product_suggestions(documents: List[Any]) -> str:
    """
    Formats product documents as product cards if they contain product data.
    
    Args:
        documents: List of document objects that may contain product information
        
    Returns:
        str: HTML-formatted product cards or empty string if no products found
    """
    if not documents:
        return ""
    
    product_cards = []
    
    for doc in documents:
        try:
            # Check if document has metadata and contains product information
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                
                # Check if this is a product document (has 'type' field set to 'product')
                if metadata.get('doc_type') == 'product':
                    # Get product code from metadata
                    product_code = metadata.get('product_id') or metadata.get('code') or metadata.get('sku', 'N/A')
                    
                    # Fetch thumb_image from product.json file
                    thumb_image = get_product_image_from_json(product_code) if product_code != 'N/A' else ''
                    
                    product_data = {
                        'code': product_code,
                        'title': metadata.get('product_title', 'Product'),
                        'thumb_image': thumb_image,
                        'url': metadata.get('product_url', '')
                    }
                    product_cards.append(format_product_card(product_data))
        except Exception as e:
            print(f"Error formatting product card: {e}")
            continue
    
    if product_cards:
        # Add a header and combine all product cards
        result = "\n\n### Suggested Products\n\n"
        result += "\n".join(product_cards)
        return result
    
    return ""