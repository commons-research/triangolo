
# %%
# We import the required libraries

# Here we import the helpers functions (see helpers.py)
from helpers import load_and_filter_from_mgf
from helpers import combinator
from helpers import aggregator
from helpers import plot_it_quick
from helpers import plot_it_interactive

# %%
# Define the filename you want to use for the outputs (html interactive plots and csv of parents/daughters/losses)

filename = 'cocaine'

# Define the path of the corresponding spectral file (mfg format)

path_to_mgf = './data/input/cocaine.mgf'

# %%
# Load the spectra defining the parsing parameters

spectras = load_and_filter_from_mgf(path=path_to_mgf, min_relative_intensity = 0.01,
            max_relative_intensity = 1, n_required=5, loss_mz_from = 10, loss_mz_to = 200)

# %%
# calculate all combinations of losses across the spectral file (no settings required)

pdl_combinations_df = combinator(spectras, 1)


# %%
# aggregate the calculated parent / daughter / loss combinations

agg_pdl_df = aggregator(pdl_combinations_df)

# %%
# output as a csv if needed

agg_pdl_df.to_csv('./data/output/' + filename + '_pdl.csv')

# %%
# quick static plot using datashader 
# just specify the input and the data which should be plotted n the x and y axis 
# >>> choose between parent, daughter or loss


plot_it_quick(agg_pdl_df, 'parent', 'loss')

# %%
# interactive plot using datashader 
# specify the input and the data which should be plotted n the x and y axis 
# >>> choose a pair between parent, daughter or loss
# you can additionally specify the resolution of the plot  
# >>> choose between parent, daughter or loss

plot_it_interactive(df=agg_pdl_df,
                    x='parent',
                    y='daughter',
                    plot_width_value=1000,
                    plot_height_value=1000,
                    output_path='./data/output/',
                    filename=filename
                    )
