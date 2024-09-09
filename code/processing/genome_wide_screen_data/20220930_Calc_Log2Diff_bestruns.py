import numpy as np
import pandas as pd
import gzip
import glob
import os
import sys

from sklearn import preprocessing
import matplotlib.pyplot as plt

sys.path.append('../../..')

def scale_data(array, means,stds):
    return (array-means)/stds


#################################
#################################
# The aim of this file is to convert the raw sequence counts
# into log2-foldchange data for each galvanotaxis experiment.
#################################
#################################

# load in sgRNA library IDs
df_IDs = pd.read_csv('../../../data/seq/Sanson_Dolc_CRISPRI_IDs_libA.csv')
df_IDs.columns = ['sgRNA', 'annotated_gene_symbol', 'annotated_gene_ID']

df = df_IDs.copy()
exp_list_set1 = ['NB217', 'NB215', 'NB216', 'NB214', 'NB211', 'NB212',
                'NB220', 'NB218', 'NB219', 'NB227', 'NB224',
                'NB226', 'NB233', 'NB231', 'NB232', 'NB234']
dirname = '/Volumes/Belliveau_RAW_3_JTgroup/Sequencing_RAW/20210810_novogene/raw_data/'

for exp in exp_list_set1:
    df_temp = pd.read_csv(dirname + exp + '/trim_seqtk_sgRNAcounts_20210810.csv')

    df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
    df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
    df_temp_notmatched[''.join(['counts_',exp])] = 0

    df_temp_matched = df_temp_matched[['sgRNA', 'index']]

    df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
    df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])

    df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')

#################################
# load in sgRNA counts
#################################
# ################## A1

df['NB233_NB224_diff'] = np.log2(df.counts_NB233 + 32) - np.log2(df.counts_NB233.median()) - (np.log2(df.counts_NB224 + 32) - np.log2(df.counts_NB224.median()))

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB233_NB224_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'top_A_2'

df_compare = df_temp.copy()


df['NB226_NB224_diff'] = np.log2(df.counts_NB226 + 32) - np.log2(df.counts_NB226.median()) - (np.log2(df.counts_NB224 + 32) - np.log2(df.counts_NB224.median()))

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB226_NB224_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
df_temp['diff'] = preprocessing.scale(df_temp['diff'])
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'bottom_A_2'

df_compare = df_compare.append(df_temp)

# ################## B1

df['NB211_NB214_diff'] = np.log2(df.counts_NB211 + 32) - np.log2(df.counts_NB211.median()) - (np.log2(df.counts_NB214 + 32) - np.log2(df.counts_NB214.median()))

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB211_NB214_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'top_B_1'

df_compare = df_compare.append(df_temp)


# ################## B3

df['NB234_NB231_diff'] = np.log2(df.counts_NB234 + 32) - np.log2(df.counts_NB234.median()) - (np.log2(df.counts_NB231 + 32) - np.log2(df.counts_NB231.median()))

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB234_NB231_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'top_B_3'

df_compare = df_compare.append(df_temp)


df['NB232_NB231_diff'] = np.log2(df.counts_NB232 + 32) - np.log2(df.counts_NB232.median()) - (np.log2(df.counts_NB231 + 32) - np.log2(df.counts_NB231.median()))

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB232_NB231_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
df_temp['diff'] = preprocessing.scale(df_temp['diff'])
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'bottom_B_3'

df_compare = df_compare.append(df_temp)

#################################
# Save collated log2fold change values to disk
#################################
df_compare.to_csv('../../../data/screen_summary/log2foldchange/20220930_screen_log2fold_diffs_galvo_sgRNA_bestruns_original_analysis.csv')

#################################
# Calculate mean values for each sgRNA and save to disk
#################################
df_mean = pd.DataFrame()
for sg, d in df_compare.groupby('sgRNA'):
    data_list = {'sgRNA' : sg,
    'annotated_gene_symbol' : d.annotated_gene_symbol.unique(),
    'diff' : d['diff'].mean()}

    df_mean = df_mean.append(data_list, ignore_index = True)

df_mean['gene'] = [i[0] for i in df_mean.annotated_gene_symbol.values]

#cleanup dataframe columns
df_mean = df_mean[['gene', 'sgRNA', 'diff']]
df_mean.columns = ['gene', 'sgRNA', 'log2fold_diff_mean']

df_mean.to_csv('../../../data/screen_summary/log2foldchange/20220930_screen_log2fold_diffs_galvo_sgRNA_means_bestruns_original_analysis.csv')
