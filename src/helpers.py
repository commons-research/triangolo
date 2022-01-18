
from matchms.importing import load_from_mgf
from matchms.filtering import add_losses
from matchms.filtering import add_precursor_mz
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_relative_intensity
import pandas as pd
import numpy as np
from itertools import combinations
import datashader as ds
import colorcet as cc
import plotly.express as px
import inspect


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

def combinator(spectras, precision):

    """Iterates over a list of matchms.spectrum objects and returns all combination of losses across a spectra
    Args:
            spectras (a list of matchms.spectrum objects): a list of matchms.spectrum objects
            precision (int): precision for the losses calculation (in number of decimals)
            r (int): will return successive r-length combinations of elements in the iterable.
    Returns:
        f : a dataframe of parent, daughter and losses
    """
    # We iterate over each spectra of the spectra ensemble and calculate 

    # We define and empty dict
    pdl_combinations_dict = {}

    for i in range(len(spectras)):
        # prec_mz = spectras[i].get("precursor_mz")
        # here we adapt the script to the latest Spikes class
        peaks_mz = spectras[i].peaks.mz
        peaks_intensities  = spectras[i].peaks.intensities
        peaks_mz = np.round(peaks_mz, precision)
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
        pdl_combinations_dict[i] = df_d
        # npa = df_d[['parent', 'loss']].to_numpy()
        # df_d['COUNT'] = 1
        # mat = df_d.pivot_table('COUNT', index='parent', columns="loss").fillna(0)
    
    pdl_combinations_df = pd.concat(pdl_combinations_dict.values())
    return pdl_combinations_df

def aggregator(pdl_combinations_df):

    """Aggregates and pivots a pdl_combinations_df (parent, daughter, loss) dataframe

    Returns:
        f : a dataframe of parent, daughter and losses
    """

    aggregated_pdl_combinations_df = pdl_combinations_df.pivot_table(columns=['parent','daughter','loss'], aggfunc='size')
    aggregated_pdl_combinations_df = pd.DataFrame(aggregated_pdl_combinations_df)
    aggregated_pdl_combinations_df.reset_index(inplace=True)
    aggregated_pdl_combinations_df.rename(columns={0: 'count'}, inplace=True)

    return aggregated_pdl_combinations_df

def plot_it_quick(df,x,y):

    """Plots a datashader scatter. Takes a dataframe as input and requires to specify which data is to be plotted on the x and y axis.

    Returns:
        A datashader scatter plot
    """

    agg = ds.Canvas().points(df, x, y)
    ds_plot = ds.tf.set_background(ds.tf.shade(agg, cmap=cc.fire), "black")
    return ds_plot


def plot_it_interactive(df,x,y,plot_width_value, plot_height_value, output_path, filename):

    """Plots and saves an interactive datashader scatter. Takes a dataframe as input and requires to specify which data is to be plotted on the x and y axis.
    Args:
            df (str): Name of the dataframe to plot
            x (str): Value to plot in the x axis
            y (str): Value to plot in the y axis
            plot_width_value (int): Resolution of the datashader plot on the x axis
            plot_height_value (int): Resolution of the datashader plot on the y axis
            output_file_path (str): Path towards the .html output
            filename (str): Filename (no extension required)

    Returns:
        A datashader scatter plot
    """
    cvs = ds.Canvas(plot_width=plot_width_value, plot_height=plot_height_value)
    agg = cvs.points(df, x, y)
    zero_mask = agg.values == 0
    agg.values = np.log10(agg.values, where=np.logical_not(zero_mask))
    agg.values[zero_mask] = np.nan
    fig = px.imshow(agg, origin='lower', labels={'color':'Log10(count)'})
    fig.update_traces(hoverongaps=False)
    fig.update_layout(coloraxis_colorbar=dict(title='Count', tickprefix='1.e'))
    fig.update_layout(title= 'Scatter of ' + x +' - '+ y +' combinations for ' + filename)
    fig.show()
    fig.write_html(output_path + '/'+ filename +'_'+ x +'_'+ y +'_'+ str(plot_width_value) +'x'+ str(plot_height_value) + '.html',
                        full_html=False,
                        include_plotlyjs='cdn')




