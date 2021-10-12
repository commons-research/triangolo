import pandas as pd
import numpy as np
from matchms.importing import load_from_mgf
from matchms.filtering import add_losses
from matchms.filtering import add_precursor_mz
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_relative_intensity

def load_and_filter_from_mgf(path, min_relative_intensity, max_relative_intensity, loss_mz_from, loss_mz_to, n_required) -> list:
    """Load and filter spectra from mgf file to prepare for MEMO matrix generation

    Returns:
        spectrums (list of matchms.spectrum): a list of matchms.spectrum objects
    """
    def apply_filters(spectrum):
        spectrum = normalize_intensities(spectrum)
        spectrum = select_by_relative_intensity(spectrum, intensity_from = min_relative_intensity, intensity_to = max_relative_intensity)
        spectrum = add_precursor_mz(spectrum)
        spectrum = add_losses(spectrum, loss_mz_from= loss_mz_from, loss_mz_to= loss_mz_to)
        spectrum = require_minimum_number_of_peaks(spectrum, n_required= n_required)
        return spectrum

    spectra_list = [apply_filters(s) for s in load_from_mgf(path)]
    spectra_list = [s for s in spectra_list if s is not None]
    return spectra_list 

def import_mzmine2_quant_table(path) -> pd.DataFrame:
    """Import feature quantification table generated from MzMine 2 and clean it

    Args:
        path (str): Path to feature quantification table

    Returns:
        quant_table (DataFrame): A cleaned MzMine2 feature quantification table
    """
    quant_table = pd.read_csv(path, sep=',')
    quant_table.set_index('row ID', inplace=True)
    quant_table = quant_table.filter(like='Peak area', axis=1)
    quant_table.rename(columns = lambda x: x.replace(' Peak area', ''), inplace=True)
    quant_table = quant_table.transpose()
    quant_table.index.name = 'filename'
    return quant_table


import numpy
from ..Spikes import Spikes
from ..typing import SpectrumType


def add_losses_full(spectrum_in: SpectrumType, loss_mz_from=0.0, loss_mz_to=1000.0) -> SpectrumType:
    """Derive losses based on precursor mass.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    loss_mz_from:
        Minimum allowed m/z value for losses. Default is 0.0.
    loss_mz_to:
        Maximum allowed m/z value for losses. Default is 1000.0.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    precursor_mz = spectrum.get("precursor_mz")
    if precursor_mz:
        assert isinstance(precursor_mz, (float, int)), ("Expected 'precursor_mz' to be a scalar number.",
                                                        "Consider applying 'add_precursor_mz' filter first.")
        peaks_mz, peaks_intensities = spectrum.peaks
        losses_mz = (precursor_mz - peaks_mz)[::-1]
        losses_intensities = peaks_intensities[::-1]
        # Add losses which are within given boundaries
        mask = numpy.where((losses_mz >= loss_mz_from)
                           & (losses_mz <= loss_mz_to))
        spectrum.losses = Spikes(mz=losses_mz[mask],
                                 intensities=losses_intensities[mask])

    return spectrum


def load_and_filter_from_mgf_full(path, min_relative_intensity, max_relative_intensity, loss_mz_from, loss_mz_to, n_required) -> list:
    """Load and filter spectra from mgf file to prepare for MEMO matrix generation

    Returns:
        spectrums (list of matchms.spectrum): a list of matchms.spectrum objects
    """
    def apply_filters(spectrum):
        spectrum = normalize_intensities(spectrum)
        spectrum = select_by_relative_intensity(spectrum, intensity_from = min_relative_intensity, intensity_to = max_relative_intensity)
        spectrum = add_precursor_mz(spectrum)
        spectrum = add_losses(spectrum, loss_mz_from= loss_mz_from, loss_mz_to= loss_mz_to)
        spectrum = require_minimum_number_of_peaks(spectrum, n_required= n_required)
        return spectrum

    spectra_list = [apply_filters(s) for s in load_from_mgf(path)]
    spectra_list = [s for s in spectra_list if s is not None]
    return spectra_list 