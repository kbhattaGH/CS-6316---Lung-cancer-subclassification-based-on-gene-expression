##
import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import os

#
class CancerClassifier:
    def __init__(self, gamma, data, components=1000):
        self.gamma = gamma
        self.components = components
        self.data = data

    # def process_data(self):
    #     #import data from processed file and extract required meta-data and train-tes split
    #     features = self.data.drop(data.drop(columns=['label','index']))
    #     labels = self.data[['index','label']]
    #     SS = StandardScaler(copy=True, with_mean=True, with_std=True)
    #     SS.fit(features)
    #     scaled_features = SS.transform(features)
    #     pca = PCA(n_components=self.components)
    #     trans_data = pca.fit_transform(scaled_features)  # need to check variance ot make sure
    #
    #     X_Train, X_Test, Y_Train, Y_Test = sk.model_selection.train_test_split()
    #
    #     return X_Train, Y_Train, X_Test, Y_Test



    def train_classifier(self):

        cls = sk.svm.SVC(gamma=self.gamma)
        cls.fit(X_Train, Y_Train)
        train_score = cls.score(X_Train, Y_Train)
        return cls, train_score

    def test_and_graph_results(self, cls, X_Test, Y_Test):
        X_predicted = cls.predict(X_Test)
        X_predicted_score = cls.score(X_predicted, Y_Test)
# #

if __name__ == "__main__":
    abspath = sys.path[0]
    data_path = os.path.join(abspath, 'data')
    features = pd.read_csv(data_path+'/demo_feature_file.csv', sep=',', index_col=0)
    labels = pd.read_csv(data_path+'/demo.csv', sep=',', index_col=0)
    data = (pd.concat([features, labels], axis=1)).reset_index()
        #cfier = CancerClassifier(f1, gamma=50, split=[80,20])
        #data = cfier.process_data()
    print(data.keys())
    print (data[['index','label']])




