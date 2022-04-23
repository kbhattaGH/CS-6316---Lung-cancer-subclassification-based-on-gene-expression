import sys 
import os
import pandas as pd

# Import Data and Functions
abspath = sys.path[0]
src_path = os.path.abspath(os.path.join(abspath, 'src'))
sys.path.append(src_path)
from dimensionality_reduction import tSNE_reduce, plot_tSNE

# re-read in data to get fresh, unaltered dataframes
data_path = os.path.join(abspath, 'data')

print("Reading data.")
feature_data1 = pd.read_csv(data_path+'/full/train_features.csv')
feature_data2 = pd.read_csv(data_path+'/full/test_features.csv')
data_labeled = pd.concat([feature_data1, feature_data2], ignore_index=True)
data_labeled.rename(columns={'samples': 'id'}, inplace=True)

print("Running tsne.")
data_reduced_tsne = tSNE_reduce(data_labeled, n_components=2)
plot_tSNE(data_reduced_tsne)