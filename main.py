##
import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import os


class CancerClassifier:
    def __init__(self, gamma, data, components=1000, test_size=0.33):
        self.gamma = gamma
        self.components = components
        self.data = data
        self.test_size = test_size

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

    def process_data(self):
        features = self.data.drop(columns=['label'])
        labels = self.data[['label']]
        OHE_labels = labels.where(labels!="adenomas_and_adenocarcinomas",other=1)
        OHE_labels = OHE_labels.where(labels=="adenomas_and_adenocarcinomas",other=0)
        X_Train, X_Test, Y_Train, Y_Test = sk.model_selection.train_test_split(features, OHE_labels, test_size=self.test_size)
        return X_Train, X_Test, Y_Train, Y_Test

    def train_classifier(self, X_Train, Y_Train):

        cls = sk.svm.SVC(gamma=self.gamma)
        cls.fit(X_Train, Y_Train)
        train_score = cls.score(X_Train, Y_Train)
        return cls, train_score

    def test_and_graph_results(self, cls, X_Test, Y_Test):
        X_predicted = cls.predict(X_Test)
        X_predicted_score = cls.score(X_predicted, Y_Test)
        return X_predicted_score

if __name__ == "__main__":

    ####IMPORT DATA AND FUNCTIONS#####
    abspath = sys.path[0]
    src_path = os.path.abspath(os.path.join(abspath, 'src'))
    sys.path.append(src_path)
    from dimensionality_reduction import pca_reduce

    data_path = os.path.join(abspath, 'data')
    feature_data = pd.read_csv(data_path+'/demo_feature_file.csv')

    feature_data.rename(columns={'samples': 'id'}, inplace=True)

    # read in labels
    labels = pd.read_csv(data_path+"/demo.csv")

    # join on id
    data_labeled = feature_data.merge(labels, on="id", how="inner")
    data_reduced = pca_reduce(data_labeled, n_components=10)

    print(data_reduced)

    # Classifier
    cfier = CancerClassifier(gamma=10, data=data_reduced, components=1000)
    X_Train, X_Test, Y_Train, Y_Test = cfier.process_data()
    cls, train_score = cfier.train_classifier(X_Train, Y_Train)
    X_predicted_score = cfier.test_and_graph_results(cls, X_Test, Y_Test)







