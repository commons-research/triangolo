import struct

def deserialize_from_custom_binary(binary_data):
    """
    Deserialize spectrum data from custom binary format.

    Parameters:
    - binary_data (bytes): Binary data to deserialize.

    Returns:
    - metadata (dict): Metadata including 'PEPMASS' and 'CHARGE'.
    - spectrum_data (list of tuples): List of (mz, intensity) tuples.
    """
    metadata = {}
    spectrum_data = []
    
    # Read metadata
    pepmass, charge = struct.unpack('fI', binary_data[:8])
    metadata['PEPMASS'] = pepmass
    metadata['CHARGE'] = charge
    
    # Read spectrum data
    data_size = len(binary_data) - 8  # Skip metadata
    data_entry_size = struct.calcsize('fI')
    for i in range(8, len(binary_data), data_entry_size):
        mz, intensity = struct.unpack('fI', binary_data[i:i+data_entry_size])
        spectrum_data.append((mz, intensity))
    
    return metadata, spectrum_data

def deserialize_from_u64(binary_data, mz_scale=10000):
    """
    Deserialize spectrum data from U64 format.

    Parameters:
    - binary_data (bytes): Binary data to deserialize.
    - mz_scale (int, optional): Scaling factor for mz values.

    Returns:
    - metadata (dict): Metadata including 'PEPMASS' and 'CHARGE'.
    - spectrum_data (list of tuples): List of (mz, intensity) tuples.
    """
    metadata = {}
    spectrum_data = []
    
    # Read metadata
    pepmass, charge = struct.unpack('fI', binary_data[:8])
    metadata['PEPMASS'] = pepmass
    metadata['CHARGE'] = charge
    
    # Read spectrum data
    data_size = len(binary_data) - 8  # Skip metadata
    data_entry_size = struct.calcsize('Q')
    for i in range(8, len(binary_data), data_entry_size):
        mz_intensity = struct.unpack('Q', binary_data[i:i+data_entry_size])[0]
        mz = mz_intensity >> 32
        intensity = mz_intensity & ((1 << 32) - 1)
        mz /= mz_scale
        spectrum_data.append((mz, intensity))
    
    return metadata, spectrum_data
