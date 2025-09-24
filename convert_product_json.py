#!/usr/bin/env python3
"""
Convert product.json to product.csv

This script uses the json_to_csv_converter module to convert the product.json file to CSV format.
"""

import os
from utils.json_to_csv_converter import convert_product_json

def main():
    # Define input and output file paths
    input_file = os.path.join('dataset', 'output', 'product.json')
    output_file = os.path.join('dataset', 'output', 'product.csv')
    
    # Convert the file
    print(f"Converting {input_file} to {output_file}...")
    convert_product_json(input_file, output_file)
    print("Conversion complete!")

if __name__ == "__main__":
    main()
