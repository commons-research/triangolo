# To add a new cell, type '# %%'

# %%
import pandas as pd
import numpy as np
import memo_ms as memo
import plotly.express as px
import os
from itertools import combinations
from matchms.importing import load_from_mgf
from matchms.filtering import add_losses
from matchms.filtering import add_precursor_mz
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_relative_intensity

import datashader as ds


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


path_to_mgf = '/Users/pma/Dropbox/Research_UNIGE/Projets/Ongoing/sylvian-cretton/Erythroxylum_project/Fresh_Erythro/coca.mgf'

path_to_mgf = '/Users/pma/Dropbox/Research_UNIGE/Projets/Ongoing/thilo-kohler/TK_mutants_PA14_vs_wspF/TK_mutants_PA14_vs_wspF_spectra.mgf'

# outfile = 'data/cocaine1000.mgf.html'
outfile = 'data/pa.mgf.html'

spectras = load_and_filter_from_mgf(path=path_to_mgf, min_relative_intensity = 0.01,
            max_relative_intensity = 1, n_required=5, loss_mz_from = 10, loss_mz_to = 200)


# We iterate over each spectra of the spectra ensemble and calculate all combination of losses across a spectra

# We define and empty dict
f = {}

for i in range(len(spectras)):
    # prec_mz = spectras[i].get("precursor_mz")
    # here we adapt the script to the latest Spikes class
    peaks_mz = spectras[i].peaks.mz
    peaks_intensities  = spectras[i].peaks.intensities
    peaks_mz = np.round(peaks_mz, 2)
    d = []
    for a, b in combinations(peaks_mz, 2):
        # print(b, a , abs(a - b))
        d.append(
                {
                    'parent': b,
                    'daughter': a,
                    'loss':  abs(a - b)
                }
        )
    df_d = pd.DataFrame(d)
    df_d = df_d[df_d['loss'] >= 1 ]
    # f.append(df_d)
    f[i] = df_d
    # npa = df_d[['parent', 'loss']].to_numpy()
    # df_d['COUNT'] = 1
    # mat = df_d.pivot_table('COUNT', index='parent', columns="loss").fillna(0)


full_pl = pd.concat(f.values())



full_counted = full_pl.pivot_table(columns=['parent','loss'], aggfunc='size')
full_counted = pd.DataFrame(full_counted)
full_counted.reset_index(inplace=True)
full_counted.rename(columns={0: 'count'}, inplace=True)




df = full_counted

cvs = ds.Canvas(plot_width=10000, plot_height=1000)
agg = cvs.points(df, 'parent', 'loss')
zero_mask = agg.values == 0
agg.values = np.log10(agg.values, where=np.logical_not(zero_mask))
agg.values[zero_mask] = np.nan
fig = px.imshow(agg, origin='lower', labels={'color':'Log10(count)'})
fig.update_traces(hoverongaps=False)
fig.update_layout(coloraxis_colorbar=dict(title='Count', tickprefix='1.e'))
fig.show()
fig.write_html(outfile,
                    full_html=False,
                    include_plotlyjs='cdn')

df.to_csv('memo_mn_pseudo.csv')