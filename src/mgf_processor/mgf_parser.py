from typing import List, Tuple


# def parse_mgf(file_path):
#     """
#     Parse MGF file containing multiple spectra.

#     Parameters:
#     - file_path (str): Path to the MGF file.

#     Returns:
#     - spectra_data (list of dicts): List of dictionaries containing metadata and spectrum data for each spectrum.
#     """
#     spectra_data = []
#     current_spectrum = None
    
#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line.startswith('BEGIN IONS'):
#                 if current_spectrum:
#                     spectra_data.append(current_spectrum)
#                 current_spectrum = {'metadata': {}, 'spectrum_data': []}
#             elif line.startswith('END IONS'):
#                 if current_spectrum:
#                     spectra_data.append(current_spectrum)
#                     current_spectrum = None
#             elif current_spectrum is not None:
#                 parts = line.split('=')
#                 if len(parts) == 2:
#                     key, value = parts
#                     current_spectrum['metadata'][key] = value
#                 elif line.strip() and not line.startswith(('SCANS', 'RTINSECONDS', 'MSLEVEL', 'FEATURE_ID', 'MERGED_STATS')):
#                     mz, intensity = map(float, line.split())
#                     current_spectrum['spectrum_data'].append((mz, intensity))
    
#     return spectra_data


def parse_mgf(mgf_content: str) -> List[Tuple[float, int]]:
    spectrum_data = []
    mz_max = -float('inf')
    intensity_max = -float('inf')
    
    lines = mgf_content.strip().split('\n')
    for line in lines:
        if line.strip() and not line.startswith(('BEGIN', 'END', 'PEPMASS', 'CHARGE', 'RTINSECONDS', 'SCANS', 'MSLEVEL', 'FEATURE_ID', 'MERGED_STATS')):
            parts = line.split()
            mz = float(parts[0])
            intensity = int(float(parts[1]))
            spectrum_data.append((mz, intensity))
            mz_max = max(mz_max, mz)
            intensity_max = max(intensity_max, intensity)

    return spectrum_data, mz_max, intensity_max

