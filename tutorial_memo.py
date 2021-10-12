# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # MEMO Tutorial on the Qemistree Evaluation Dataset
# %% [markdown]
# In this tutorial, we will use the Qemistree published dataset (https://doi.org/10.1038/s41589-020-00677-3) to apply the MS2 BasEd SaMple VectOrization (MEMO) method
# 
# The dataset is constitued of 2 fecal samples, 1 tomato sample and 1 plasma sample, plus different binary/quaternary mixtures of these four samples. Samples were profiled in UHPLC-MS/MS (Q-Exactive spectrometer) using 2 different LC-methods. Each sample was acquired in triplicates using each LC-method (see Qemistree paper for details).
# %% [markdown]
# ## First we import the needed packages
# Be sure to have first installed memo using 'pip install memo' within the memo environment. 
# Also make sure to launch this notebook using the memo conda environement.

# %%
import pandas as pd
import numpy as np
import memo_ms as memo
import plotly.express as px
import os
import xgboost


# %%
# this cell is tagged 'parameters'
fake_parameter = 'foo'

# %% [markdown]
# With this step we import metadata:

# %%

def conditions(df_meta):
    if ((df_meta['Proportion_Fecal_1']>0) & (df_meta['Proportion_Fecal_2']==0)& (df_meta['Proportion_Tomato']==0) & (df_meta['Proportion_NIST_1950_SRM']==0)):
        return 'Fecal_1'
    if ((df_meta['Proportion_Fecal_1']==0) & (df_meta['Proportion_Fecal_2']>0)& (df_meta['Proportion_Tomato']==0) & (df_meta['Proportion_NIST_1950_SRM']==0)):
        return 'Fecal_2'
    if ((df_meta['Proportion_Fecal_1']==0) & (df_meta['Proportion_Fecal_2']==0)& (df_meta['Proportion_Tomato']>0) & (df_meta['Proportion_NIST_1950_SRM']==0)):
        return 'Tomato'
    if ((df_meta['Proportion_Fecal_1']==0) & (df_meta['Proportion_Fecal_2']==0)& (df_meta['Proportion_Tomato']==0) & (df_meta['Proportion_NIST_1950_SRM']>0)):
        return 'Plasma'
    if ((df_meta['Proportion_Fecal_1']>0) & (df_meta['Proportion_Fecal_2']>0)& (df_meta['Proportion_Tomato']==0) & (df_meta['Proportion_NIST_1950_SRM']==0)):
        return 'Fecal_1 + Fecal_2'
    if ((df_meta['Proportion_Fecal_1']>0) & (df_meta['Proportion_Fecal_2']==0)& (df_meta['Proportion_Tomato']>0) & (df_meta['Proportion_NIST_1950_SRM']==0)):
        return 'Fecal_1 + Tomato'
    if ((df_meta['Proportion_Fecal_1']>0) & (df_meta['Proportion_Fecal_2']==0)& (df_meta['Proportion_Tomato']==0) & (df_meta['Proportion_NIST_1950_SRM']>0)):
        return 'Fecal_1 + Plasma'
    if ((df_meta['Proportion_Fecal_1']==0) & (df_meta['Proportion_Fecal_2']>0)& (df_meta['Proportion_Tomato']>0) & (df_meta['Proportion_NIST_1950_SRM']==0)):
        return 'Fecal_2 + Tomato'
    if ((df_meta['Proportion_Fecal_1']==0) & (df_meta['Proportion_Fecal_2']>0)& (df_meta['Proportion_Tomato']==0) & (df_meta['Proportion_NIST_1950_SRM']>0)):
        return 'Fecal_2 + Plasma'
    if ((df_meta['Proportion_Fecal_1']==0) & (df_meta['Proportion_Fecal_2']==0)& (df_meta['Proportion_Tomato']>0) & (df_meta['Proportion_NIST_1950_SRM']>0)):
        return 'Tomato + Plasma'
    if ((df_meta['Proportion_Fecal_1']>0) & (df_meta['Proportion_Fecal_2']>0)& (df_meta['Proportion_Tomato']>0) & (df_meta['Proportion_NIST_1950_SRM']>0)):
        return 'Fecal_1 + Fecal_2 + Tomato + Plasma' 
    else:
        return 'What is it? :)'


df_meta = pd.read_csv("data/1901_gradient_benchmarking_dataset_v4_sample_metadata.txt", sep='\t')
df_meta['Samplename'] = df_meta['Samplename'].str[:-6]
df_meta['Samplename'] = df_meta['Samplename'].str.replace('BLANK_', 'BLANK')
df_meta = df_meta[['Filename', 'Experiment', 'Samplename', 'Triplicate_number', 'Proportion_Fecal_1', 'Proportion_Fecal_2', 'Proportion_Tomato', 'Proportion_NIST_1950_SRM']]
df_meta['contains'] = df_meta.apply(conditions, axis=1)
df_meta['instrument'] = np.where(df_meta['Samplename'].str.contains('qTOF'), 'qTOF', 'QE')
df_meta['blank_qc'] = np.where(df_meta['Samplename'].str.contains('blank|qcmix', case = False), 'yes', 'no')
df_meta

# Here we directly fetch data from GNPS

# gnps_job_id = '3197f70bed224f9ba6f59f62906839e9'
# input_folder = 'data/local'


# path_to_folder = os.path.expanduser(os.path.join(input_folder , gnps_job_id))
# path_to_file = os.path.expanduser(os.path.join(input_folder , gnps_job_id + '.zip'))

# download_gnps_job = True 

# # Downloading GNPS files
# if download_gnps_job == True:

#     print('''
#     Fetching the GNPS job: '''
#     + gnps_job_id
#     )

#     job_url_zip = "https://gnps.ucsd.edu/ProteoSAFe/DownloadResult?task="+gnps_job_id+"&view=download_cytoscape_data"
#     print(job_url_zip)

#     cmd = 'curl -d "" '+job_url_zip+' -o '+path_to_file+ ' --create-dirs'
#     subprocess.call(shlex.split(cmd))

#     with zipfile.ZipFile(path_to_file, 'r') as zip_ref:
#         zip_ref.extractall(path_to_folder)

#     # We finally remove the zip file
#     os.remove(path_to_file)

#     print('''
#     Job successfully downloaded: results are in: '''
#     + path_to_folder
#     )

# %% [markdown]
# ## Import feature_quant table
# To compute the MEMO matrix of the dataset, we need the table reporting presence/absence of each metabolite in each sample. This information is in the quant table and we create a memo.FeatureTable dataclass object to load it.

# %%
feat_table_qe = memo.FeatureTable(path="data/quantification_table-00000.csv")
feat_table_qe.quant_table

# %% [markdown]
# ## Import spectra
# Since MEMO relies on the occurence of MS2 fragments/losses in samples to compare them, we obviously need to importthe features' fragmentation spectra. Losses are computed and spectra translated into documents. Store in memo.SpectraDocuments dataclass object.

# %%
spectra_qe = memo.SpectraDocuments(path="data/qemistree_specs_ms.mgf", min_relative_intensity = 0.01,
            max_relative_intensity = 1, min_peaks_required=5, losses_from = 10, losses_to = 200, n_decimals = 2)


spectra_qe.document['documents'][1]


spectra_qe = memo.SpectraDocuments(path="data/qemistree_specs_ms.mgf", min_relative_intensity = 0.01,
            max_relative_intensity = 1, min_peaks_required=5, losses_from = 10, losses_to = 200, n_decimals = 2)

spectra_qe.document.get("documents")

import pandas as pd
import numpy as np
from matchms.importing import load_from_mgf
from matchms.filtering import add_losses
from matchms.filtering import add_precursor_mz
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_relative_intensity

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

spectras = load_and_filter_from_mgf_full(path="data/qemistree_specs_ms.mgf", min_relative_intensity = 0.01,
            max_relative_intensity = 1, n_required=5, loss_mz_from = 10, loss_mz_to = 200)

spectras[1].get("precursor_mz")

spectras[1].peaks

peaks_mz, peaks_intensities = spectras[1].peaks

a = np.array([1,3,7,11,13,17,19])
a = peaks_mz
d = 1
a[d:] - a[:-d]

np.diff(peaks_mz)

Could be an expesnive one to compute !!


https://www.geeksforgeeks.org/print-distinct-absolute-differences-of-all-possible-pairs-from-a-given-array/






# %% [markdown]
# ## Generation of MEMO matrix
# 
# Using the generated documents and the quant table, we can now obtain the MEMO matrix. The MEMO matrix is stored in the MemoContainer object, along with the feature table and the documents

# %%
memo_qe = memo.MemoContainer(feat_table_qe, spectra_qe)
memo_qe.memo_matrix


# %%
memo_qe.filter_matrix(matrix_to_use='memo_matrix', samples_pattern='blank', max_occurence=100)
memo_qe.filter_matrix(matrix_to_use='feature_matrix', samples_pattern='blank', max_occurence=100)
memo_qe.filtered_memo_matrix



# %% [markdown]
# ## Plotting
# 
# We can now use the MEMO matrix to generate the PCoA of our samples

# %%

memo.plot_pcoa_2d(
    matrix= memo_qe.filtered_memo_matrix,
    df_metadata=df_meta,
    metric= 'braycurtis',
    filename_col = 'Filename',
    group_col='contains',
    norm = False,
    scaling= False,
    pc_to_plot = [1,2]
)


# %%
memo.plot_pcoa_3d(
    matrix= memo_qe.filtered_memo_matrix,
    df_metadata=df_meta,
    metric= 'braycurtis',
    filename_col = 'Filename',
    group_col='contains',
    norm = False,
    scaling= False,
    pc_to_plot = [1,2,3]
)


# %%
df_meta_sub = df_meta[df_meta['Triplicate_number'] == 1]

memo.plot_heatmap(
    matrix= memo_qe.filtered_memo_matrix,
    df_metadata=df_meta_sub,
    filename_col = 'Filename',
    group_col = 'contains',
    plotly_discrete_cm = px.colors.qualitative.Plotly,
    linkage_method='ward',
    linkage_metric = 'euclidean',
    heatmap_metric = 'braycurtis',
    norm = False,
    scaling= False
)


# %%
memo.plot_hca(
    matrix= memo_qe.filtered_memo_matrix,
    df_metadata=df_meta_sub,
    filename_col = 'Filename',
    group_col = 'contains',
    linkage_method='ward',
    linkage_metric = 'euclidean',
    norm = False,
    scaling= False
)

# %% [markdown]
# ## Merge MEMO matrix from different MzMine projects
# %% [markdown]
# First, we load as before the spectra and feature matrix to generate the memo matrix of the different projects to merge.
# 
# In this case, we will once again use the Qemistree dataset and compare for the same samples data aquired on a QToF with some on a Q-Exactive (Orbitrap).

# %%
# Generate MEMO matrix for QE data

feat_table_qe = memo.FeatureTable(path="data/qe_qtof_coanalysis/qe_quant_nogapF.csv")
spectra_qe = memo.SpectraDocuments(path="data/qe_qtof_coanalysis/qe_spectra_nogapF.mgf", min_relative_intensity = 0.01,
            max_relative_intensity = 1, min_peaks_required=5, losses_from = 10, losses_to = 200, n_decimals = 2)
memo_qe = memo.MemoContainer(feat_table_qe, spectra_qe)


# %%
# Generate MEMO matrix for QToF data

feat_table_qtof = memo.FeatureTable(path="data/qe_qtof_coanalysis/qtof_quant_nogapF.csv")
spectra_qtof = memo.SpectraDocuments(path="data/qe_qtof_coanalysis/qtof_spectra_nogapF.mgf", min_relative_intensity = 0.01,
            max_relative_intensity = 1, min_peaks_required=5, losses_from = 10, losses_to = 200, n_decimals = 2)
memo_qtof = memo.MemoContainer(feat_table_qtof, spectra_qtof)


# %%
memo_qe.filter_matrix(matrix_to_use='memo_matrix', samples_pattern='blank', max_occurence=100)
memo_qtof.filter_matrix(matrix_to_use='memo_matrix', samples_pattern='blank', max_occurence=100)

# %% [markdown]
# Now, let's merge our 2 MEMO matrix. There is one paramater, drop_not_in_common, to decide wether to keep only shared peaks/losses or all of them.

# %%
memo_merged = memo_qe.merge_memo(memo_qtof, left = 'filtered_memo_matrix', right= 'filtered_memo_matrix', drop_not_in_common=True)

# %% [markdown]
# PCoA plot of the first 2 components show that the main parameter to discriminate our samples is the Instrument

# %%

memo.plot_pcoa_2d(
    matrix= memo_merged.memo_matrix,
    df_metadata=df_meta,
    metric= 'braycurtis',
    filename_col = 'Filename',
    group_col='contains',
    norm = True,
    scaling= False,
    pc_to_plot = [1,2]
)

# %% [markdown]
# But when looking at component 2 and 3, we see that it is possible to cluster chemically related samples using this approach.

# %%

memo.plot_pcoa_2d(
    matrix= memo_merged.memo_matrix,
    df_metadata=df_meta,
    metric= 'braycurtis',
    filename_col = 'Filename',
    group_col='contains',
    norm = True,
    scaling= False,
    pc_to_plot = [2,3]
)


# %%
# here we start messing around with the ml shit

# first we concatenate the memo matrix and the metadata

df = memo_qe.filtered_memo_matrix

memo_meta = pd.merge(df, df_meta, left_on='filename',
                        right_on='Filename', how='left')

memo_meta.to_csv('data/memo_meta.csv', index= False)


def stratified_split(df, target, val_percent=0.2):
    '''
    Function to split a dataframe into train and validation sets, while preserving the ratio of the labels in the target variable
    Inputs:
    - df, the dataframe
    - target, the target variable
    - val_percent, the percentage of validation samples, default 0.2
    Outputs:
    - train_idxs, the indices of the training dataset
    - val_idxs, the indices of the validation dataset
    '''
    classes=list(df[target].unique())
    train_idxs, val_idxs = [], []
    for c in classes:
        idx=list(df[df[target]==c].index)
        np.random.shuffle(idx)
        val_size=int(len(idx)*val_percent)
        val_idxs+=idx[:val_size]
        train_idxs+=idx[val_size:]
    return train_idxs, val_idxs

df = memo_meta

df = df[df.blank_qc == 'no']

train_idxs, val_idxs = stratified_split(df, 'contains', val_percent=0.25)

val_idxs, test_idxs = stratified_split(df[df.index.isin(val_idxs)], 'contains', val_percent=0.5)


def test_stratified(df, col):
    '''
    Analyzes the ratio of different classes in a categorical variable within a dataframe
    Inputs:
    - dataframe
    - categorical column to be analyzed
    Returns: None
    '''
    classes=list(df[col].unique())
    
    for c in classes:
        print(f'Proportion of records with {c}: {len(df[df[col]==c])*1./len(df):0.2} ({len(df[df[col]==c])} / {len(df)})')

print('---------- STRATIFIED SAMPLING REPORT ----------')
print('-------- Label proportions in FULL data --------')
test_stratified(df, 'contains')
print('-------- Label proportions in TRAIN data --------')
test_stratified(df[df.index.isin(train_idxs)], 'contains')
print('------ Label proportions in VALIDATION data -----')
test_stratified(df[df.index.isin(val_idxs)], 'contains')
print('-------- Label proportions in TEST data ---------')
test_stratified(df[df.index.isin(test_idxs)], 'contains')


train_df = df[df.index.isin(train_idxs)]
X_train = train_df.loc[:, df.columns.str.startswith('peak')].values
Y_train = train_df[['contains']].values
print('Retrieved Training Data')
val_df = df[df.index.isin(val_idxs)]
X_val = val_df.loc[:, df.columns.str.startswith('peak')].values
Y_val = val_df[['contains']].values
print('Retrieved Validation Data')
test_df = df[df.index.isin(test_idxs)]
X_test = test_df.loc[:, df.columns.str.startswith('peak')].values
Y_test = test_df[['contains']].values
print('Retrieved Test Data')



import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
#store data, all in numpy arrays
training_data = {'X_train':X_train,'Y_train':Y_train,
                'X_val': X_val,'Y_val':Y_val,
                'X_test': X_test,'Y_test':Y_test}


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_jobs=None,random_state=27,
                       verbose=1)
clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))

predicted_labels = clf.predict(training_data['X_test'])

accuracy_score(training_data['Y_test'], predicted_labels)

params = {
    'n_estimators'      : range(100,500,50),
    'max_depth'         : [8, 9, 10, 11, 12],
    'max_features': ['auto'],
    'criterion' :['gini']
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch1 = GridSearchCV(estimator = clf, param_grid = params, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch1.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))

def getTrainScores(gs):
    results = {}
    runs = 0
    for x,y in zip(list(gs.cv_results_['mean_test_score']), gs.cv_results_['params']):
        results[runs] = 'mean:' + str(x) + 'params' + str(y)
        runs += 1
    best = {'best_mean': gs.best_score_, "best_param":gs.best_params_}
    return results, best

getTrainScores(gsearch1)


clf2 = gsearch1.best_estimator_

params1 = {
    'n_estimators'      : range(200,300,10),
    'max_depth'         : [11, 12,13]
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch2 = GridSearchCV(estimator = clf2, param_grid = params1, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch2.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))

getTrainScores(gsearch2)

clf3 = gsearch2.best_estimator_

params2 = {
    'n_estimators'      : range(200,220,5),
    'max_depth'         : [13,14,15]
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch3 = GridSearchCV(estimator = clf3, param_grid = params2, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch3.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))

getTrainScores(gsearch3)


clf4 = gsearch3.best_estimator_

params3 = {
    'max_depth'         : range(14,20,1)
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch4 = GridSearchCV(estimator = clf4, param_grid = params3, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch4.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))


getTrainScores(gsearch4)


clf5 = gsearch4.best_estimator_

params4 = {
    'max_depth'         : range(19,50,2)
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch5 = GridSearchCV(estimator = clf5, param_grid = params4, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch5.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))


getTrainScores(gsearch5)

clf6 = gsearch5.best_estimator_

params5 = {
    'max_depth'         : [24,25,26]
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch6 = GridSearchCV(estimator = clf6, param_grid = params5, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch6.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))

getTrainScores(gsearch6)

final_clf = gsearch6.best_estimator_

final_clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
predicted_labels = final_clf.predict(training_data['X_test'])
train_pred = final_clf.predict(training_data['X_train'])
print('Train Accuracy:'+str(accuracy_score(training_data['Y_train'], train_pred)))
print('Train F1-Score(Micro):'+str(f1_score(training_data['Y_train'], train_pred,average='micro')))
print('------')
print('Test Accuracy:'+str(accuracy_score(training_data['Y_test'], predicted_labels)))
print('Test F1-Score(Micro):'+str(f1_score(training_data['Y_test'], predicted_labels,average='micro')))

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")


import xgboost as xgb
import matplotlib.pyplot as plt

features = [col for col in df if col.startswith('peak')]
features

#allow logloss and classification error plots for each iteraetion of xgb model
def plot_compare(metrics,eval_results,epochs):
    for m in metrics:
        test_score = eval_results['val'][m]
        train_score = eval_results['train'][m]
        rang = range(0, epochs)
        plt.rcParams["figure.figsize"] = [6,6]
        plt.plot(rang, test_score,"c", label="Val")
        plt.plot(rang, train_score,"orange", label="Train")
        title_name = m + " plot"
        plt.title(title_name)
        plt.xlabel('Iterations')
        plt.ylabel(m)
        lgd = plt.legend()
        plt.show()
        
def fitXgb(sk_model, training_data=training_data,epochs=300):
    print('Fitting model...')
    sk_model.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
    print('Fitting done!')
    train = xgb.DMatrix(training_data['X_train'], label=training_data['Y_train'])
    val = xgb.DMatrix(training_data['X_val'], label=training_data['Y_val'])
    params = sk_model.get_xgb_params()
    metrics = ['mlogloss','merror']
    params['eval_metric'] = metrics
    store = {}
    evallist = [(val, 'val'),(train,'train')]
    xgb_model = xgb.train(params, train, epochs, evallist,evals_result=store,verbose_eval=100)
    print('-- Model Report --')
    print('XGBoost Accuracy: '+str(accuracy_score(sk_model.predict(training_data['X_test']), training_data['Y_test'])))
    print('XGBoost F1-Score (Micro): '+str(f1_score(sk_model.predict(training_data['X_test']),training_data['Y_test'],average='micro')))
    plot_compare(metrics,store,epochs)
    features = [col for col in df if col.startswith('peak')]
    f, ax = plt.subplots(figsize=(10,5))
    plot = sns.barplot(x=features, y=sk_model.feature_importances_)
    ax.set_title('Feature Importance')
    plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
    plt.show()



f, ax = plt.subplots(figsize=(10,5))
plot = sns.barplot(x=features, y=final_clf.feature_importances_)
ax.set_title('Feature Importance')
plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
plt.show()



from xgboost.sklearn import XGBClassifier
#initial model
xgb1 = XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    nthread=4,
                    num_class=9,
                    seed=27)



fitXgb(xgb1, training_data)

