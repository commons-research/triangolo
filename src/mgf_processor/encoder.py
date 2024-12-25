import struct
from typing import List, Tuple

def serialize_to_custom_binary(spectrum_data: List[Tuple[float, int]], mz_max: float, intensity_max: int) -> bytes:
    binary_data = struct.pack('I', len(spectrum_data))  # Number of data points
    
    for mz, intensity in spectrum_data:
        mz_scaled = int((mz / mz_max) * (2**32 - 1))
        intensity_scaled = int((intensity / intensity_max) * (2**32 - 1))
        binary_data += struct.pack('II', mz_scaled, intensity_scaled)
    
    return binary_data


# def serialize_to_custom_binary(metadata, spectrum_data):
#     """
#     Serialize spectrum data and metadata to custom binary format.

#     Parameters:
#     - metadata (dict): Metadata for the spectrum.
#     - spectrum_data (list of tuples): List of tuples containing (mz, intensity) pairs.

#     Returns:
#     - binary_data (bytes): Binary representation of the spectrum data.
#     """
#     binary_data = bytearray()

#     # Pack metadata
#     binary_data += struct.pack('f', float(metadata['PEPMASS']))
    
#     # Extract charge value
#     charge_value = metadata.get('CHARGE', '1+').strip()  # Use default value '1+' if CHARGE is missing or empty
    
#     # Check if charge value is not empty
#     if charge_value:
#         # Remove '+' from charge before converting to int
#         binary_data += struct.pack('i', int(charge_value.strip('+')))
#     else:
#         # Use default charge value '1+'
#         binary_data += struct.pack('i', 1)


#     # Pack spectrum data
#     for mz, intensity in spectrum_data:
#         binary_data += struct.pack('ff', mz, intensity)

#     return binary_data



# def serialize_to_u64(metadata, spectrum_data):
#     """
#     Serialize spectrum data and metadata to U64 format.

#     Parameters:
#     - metadata (dict): Metadata for the spectrum.
#     - spectrum_data (list of tuples): List of tuples containing (mz, intensity) pairs.

#     Returns:
#     - u64_data (bytes): U64 representation of the spectrum data.
#     """
#     # Determine scale factors for mz and intensity
#     mz_max = max(mz for mz, _ in spectrum_data)
#     intensity_max = max(intensity for _, intensity in spectrum_data)

#     mz_scale = (2**32 - 1) / mz_max
#     intensity_scale = (2**32 - 1) / intensity_max

#         # Extract charge value
#     charge_value = metadata.get('CHARGE', '1+').strip()  # Use default value '1+' if CHARGE is missing or empty

#     # Pack metadata
#     pepmass_float = float(metadata['PEPMASS'])
#     charge_int = int(charge_value.strip('+'))  # Remove '+' from charge before converting to int
#     metadata_bytes = struct.pack('fI', pepmass_float, charge_int)

#     # Pack spectrum data
#     spectrum_bytes = b""
#     for mz, intensity in spectrum_data:
#         mz_scaled = int(mz * mz_scale)
#         intensity_scaled = int(intensity * intensity_scale)
#         spectrum_bytes += struct.pack('QQ', mz_scaled, intensity_scaled)

#     # Combine metadata and spectrum data
#     u64_data = metadata_bytes + spectrum_bytes

#     return u64_data
