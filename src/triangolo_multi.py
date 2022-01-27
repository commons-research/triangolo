
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

filename = 'erythroxylum_coca'

# Define the path of the corresponding spectral file (mfg format)

path_to_mgf = '../data/input/e_coca.mgf'
path_to_mgf2 = '../data/input/e_plowmanianum.mgf'
path_to_mgf3 = '../data/input/cocaine.mgf'

mgfs_pathes = [path_to_mgf, path_to_mgf2, path_to_mgf3]

# %%
# Load the spectra defining the parsing parameters

dico_spectral = {}

for i in mgfs_pathes:
    dico_spectral[i] = load_and_filter_from_mgf(path=i, min_relative_intensity = 0.05,
            max_relative_intensity = 1, n_required=5, loss_mz_from = 10, loss_mz_to = 200)



# %%
# calculate all combinations of losses across the spectral file (no settings required)


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
        feature_id  = spectras[i].get('feature_id')
        peaks_mz = np.round(peaks_mz, precision)
        d = []
        for a, b in combinations(peaks_mz, 2):
            # print(b, a , abs(a - b))
            d.append(
                    {
                        'parent': b,
                        'daughter': a,
                        'loss':  abs(a - b),
                        'feature_id': feature_id
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
    

## 
ls = {}

for k, v in dico_spectral.items():
    ls[k] = combinator(v, 2)

appended_data = pd.concat(ls)
appended_data.reset_index(inplace = True)

# we drop some useless columns

appended_data.drop('level_1', axis = 1, inplace =True)


appended_data['source'] = appended_data["level_0"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])



basename = os.path.basename(filepath)



import collections

df_test = result.groupby(['parent','daughter','loss'])['source'].apply(list).reset_index()
df_test['count'] = df_test['source'].str.len()
collections.Counter(df_test['source'])


df_test['freq'] = df_test['source'].apply(lambda x: dict(collections.Counter(x)))


df_test['freq'].apply(pd.Series)

df_test = pd.concat([df_test.drop('freq', axis=1), pd.json_normalize(df_test['freq'])], axis=1)

df_test = df_test.drop('source', axis=1)



pd.json_normalize(df_test['freq'])

df_test = df_test[df_test['count'] >= 5]


df_test = df_test.assign(feature_id=df_test.feature_id.map(lambda x: '|'.join(map(str, x))))
df_test = df_test[df_test['count'] >= 5]
df_test['source'] = 'e_coca'

# %%
# aggregate the calculated parent / daughter / loss combinations

agg_pdl_df = aggregator(pdl_combinations_df)


df_test = pdl_combinations_df.groupby(['parent','daughter','loss'])['feature_id'].apply(list).reset_index()
df_test['count'] = df_test['feature_id'].str.len()
df_test = df_test.assign(feature_id=df_test.feature_id.map(lambda x: '|'.join(map(str, x))))
df_test = df_test[df_test['count'] >= 5]
df_test['source'] = 'e_coca'


aggregated_pdl_combinations_df = pdl_combinations_df.pivot_table(index='feature_id', columns=['parent','daughter','loss'], aggfunc='size')
aggregated_pdl_combinations_df = pd.DataFrame(aggregated_pdl_combinations_df)
aggregated_pdl_combinations_df.reset_index(inplace=True)
aggregated_pdl_combinations_df.rename(columns={0: 'count'}, inplace=True)

agg_pdl_df = agg_pdl_df[agg_pdl_df['count'] >= 5]

agg_pdl_df.info()


# %%
# output as a csv if needed

df_test.to_csv('../data/output/' + filename + '_pdl_featured_agg.csv', float_format='%g', index = None )

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
                    y='loss',
                    plot_width_value=10000,
                    plot_height_value=5000,
                    output_path='../data/output/',
                    filename=filename
                    )
