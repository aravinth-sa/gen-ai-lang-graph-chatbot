import re

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