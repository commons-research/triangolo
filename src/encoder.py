import base64
from zlib import compress, decompress
import numpy as np
import sys

# Sample data
spectrum_data = [
    # Add 50 data points in the form of (mz, intensity) (increasing m/z, random intensity)
    (66.6118, 4600), (67.0553, 34000), (69.0709, 16000),
    (71.0866, 15000), (79.0551, 8700), (81.0708, 54000),
    (83.0864, 12000), (85.1021, 12000), (87.1177, 12000),
    (89.1334, 12000), (91.149, 8000), (93.1647, 8000),
    (95.1803, 8000), (97.196, 8000), (99.2116, 8000)]


# Simulated MGF content (typically you would read this from an actual file)
mgf_content = """
BEGIN IONS
FEATURE_ID=1
PEPMASS=956.8639
SCANS=1
RTINSECONDS=572.053
CHARGE=1+
MSLEVEL=2
66.6118 4.6E3
67.0553 3.4E4
69.0709 1.6E4
71.0866 1.5E4
79.0551 8.7E3
81.0708 5.4E4
83.0864 1.9E4
85.1019 5.6E3
91.0550 1.2E4
93.0705 2.3E4
95.0863 4.6E4
95.6745 4.3E3
96.8815 8.3E3
97.1019 1.8E4
105.0701 5.3E3
107.0862 1.0E4
109.1020 2.1E4
111.1179 7.3E3
117.5334 4.7E3
121.1018 1.4E4
123.1174 7.9E3
133.1016 5.7E3
135.1171 1.1E4
138.6396 4.2E3
149.1331 7.7E3
163.1487 5.0E3
192.5042 4.4E3
261.2217 1.5E4
263.2379 6.7E3
397.3709 5.5E3
599.5043 7.8E4
629.5526 1.1E4
657.5867 5.0E4
659.5987 5.5E4
661.6133 4.0E4
680.6642 4.6E3
690.0441 4.4E3
939.8386 1.4E4
956.8868 9.1E3
END IONS
"""

def parse_mgf(mgf_string):
    spectrum_data = []
    for line in mgf_string.splitlines():
        if line.strip() and not line.startswith(('BEGIN', 'END', 'PEPMASS', 'CHARGE', 'RTINSECONDS', 'SCANS', 'MSLEVEL', 'FEATURE_ID')):
            parts = line.split()
            mz = float(parts[0])  # m/z values as float
            intensity = int(float(parts[1]))  # Convert scientific notation to int
            spectrum_data.append((mz, intensity))
    return spectrum_data

# Base64 Encoding & Decoding Functions
def encode_base64(data):
    # Normalize and scale data
    min_mz = min(mz for mz, _ in data)
    max_intensity = max(intensity for _, intensity in data)
    normalized_data = [(mz - min_mz, intensity / max_intensity) for mz, intensity in data]
    data_str = ','.join(f"{mz:.2f}:{intensity:.4f}" for mz, intensity in normalized_data)
    compressed_data = compress(data_str.encode('utf-8'))
    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    return encoded_data, len(encoded_data), min_mz, max_intensity

def decode_base64(encoded_data, min_mz, max_intensity):
    compressed_data = base64.b64decode(encoded_data)
    data_str = decompress(compressed_data).decode('utf-8')
    return [(float(mz) + min_mz, float(intensity) * max_intensity) for mz, intensity in 
            (pair.split(':') for pair in data_str.split(','))]

# U64 Encoding & Decoding Functions
def encode_to_u64(data):
    mz_max = max(mz for mz, _ in data)
    intensity_max = max(intensity for _, intensity in data)
    encoded_data = []
    for mz, intensity in data:
        mz_scaled = int((mz / mz_max) * (2**32 - 1))
        intensity_scaled = int((intensity / intensity_max) * (2**32 - 1))
        encoded_data.append((mz_scaled << 32) | intensity_scaled)
    return encoded_data, 8 * len(encoded_data), mz_max, intensity_max

def decode_from_u64(encoded_data, mz_max, intensity_max):
    decoded_data = []
    for encoded_value in encoded_data:
        mz_scaled = (encoded_value >> 32) & (2**32 - 1)
        intensity_scaled = encoded_value & (2**32 - 1)
        mz = (mz_scaled / (2**32 - 1)) * mz_max
        intensity = (intensity_scaled / (2**32 - 1)) * intensity_max
        decoded_data.append((mz, intensity))
    return decoded_data

# Parse MGF content

spectrum_data = parse_mgf(mgf_content)

# Calculate the size of the original data
original_size = sum(sys.getsizeof(mz) + sys.getsizeof(intensity) for mz, intensity in spectrum_data) + sys.getsizeof(spectrum_data)



# Encoding and comparing sizes
encoded_base64, size_base64, min_mz, max_intensity = encode_base64(spectrum_data)
encoded_u64, size_u64, mz_max, intensity_max = encode_to_u64(spectrum_data)

decoded_base64 = decode_base64(encoded_base64, min_mz, max_intensity)
decoded_u64 = decode_from_u64(encoded_u64, mz_max, intensity_max)


print("Original Data Size:", original_size, "bytes")
print("Base64 Encoded size:", size_base64, "bytes")
print("U64 Encoded size:", size_u64, "bytes")
print("Original Base64 Decoded Data:", decoded_base64)
print("Original U64 Decoded Data:", decoded_u64)



import struct

def serialize_spectrum(spectrum, mz_scale=10000, intensity_type='H'):
    """
    Serialize a single spectrum to a custom binary format.
    
    spectrum: A dictionary with 'pepmass', 'charge', and 'data' (list of (m/z, intensity) tuples).
    mz_scale: Scale factor for converting m/z values to fixed-point.
    intensity_type: Struct format character for intensities (e.g., 'H' for unsigned short).
    """
    # Header with pepmass (float) and charge (signed byte)
    header = struct.pack('fB', spectrum['pepmass'], spectrum['charge'])
    
    # Data section
    data_bytes = b''
    for mz, intensity in spectrum['data']:
        # Convert m/z to fixed-point and pack with intensity
        mz_fixed = int(mz * mz_scale)
        data_bytes += struct.pack(f'I{intensity_type}', mz_fixed, intensity)
    
    # Return the concatenated header and data bytes
    return header + data_bytes

# Example spectrum
spectrum = {
    'pepmass': 956.8639,
    'charge': 1,
    'data': [(66.6118, 4600), (67.0553, 34000), (69.0709, 16000)]
}

# Serialize the spectrum
binary_data = serialize_spectrum(spectrum)

print(f"Serialized data size: {len(binary_data)} bytes")


def deserialize_spectrum(binary_data, mz_scale=10000, intensity_type='H'):
    """
    Deserialize a single spectrum from a custom binary format back into a readable format.
    
    binary_data: The binary string to be deserialized.
    mz_scale: Scale factor used for m/z values during serialization.
    intensity_type: Struct format character used for intensities during serialization.
    """
    # Header: pepmass (float) and charge (signed byte)
    pepmass, charge = struct.unpack('fB', binary_data[:5])
    spectrum = {'pepmass': pepmass, 'charge': charge, 'data': []}
    
    # Data section
    intensity_size = struct.calcsize(intensity_type)
    data_entry_size = 4 + intensity_size  # 4 bytes for fixed-point m/z, size of intensity type
    data_section = binary_data[5:]  # Skip header
    
    for i in range(0, len(data_section), data_entry_size):
        mz_fixed, intensity = struct.unpack(f'I{intensity_type}', data_section[i:i+data_entry_size])
        mz = mz_fixed / mz_scale  # Convert back from fixed-point
        spectrum['data'].append((mz, intensity))
    
    return spectrum

# Using the serialized data from the previous example
deserialized_spectrum = deserialize_spectrum(binary_data)

print("Deserialized Spectrum:", deserialized_spectrum)
