from mgf_parser import parse_mgf
from encoder import serialize_to_custom_binary
import os

def main():
    with open('data/input/blank_one.mgf', 'r') as file:
        mgf_content = file.read()

    spectrum_data, mz_max, intensity_max = parse_mgf(mgf_content)
    
    original_data_size = os.path.getsize('data/input/blank_one.mgf')
    custom_binary_data = serialize_to_custom_binary(spectrum_data, mz_max, intensity_max)
    
    encoded_size = len(custom_binary_data)

    print("Size of original MGF file:", original_data_size, "bytes")
    print("Custom Binary Encoded Size:", encoded_size, "bytes")

if __name__ == "__main__":
    main()
