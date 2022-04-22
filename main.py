import sklearn as sk
import pandas as pd
import sys
import os
import numpy as np


class CancerClassifier:
    def __init__(self, gamma, data, test_size=0.33):
        self.gamma = gamma
        self.data = data
        self.test_size = test_size

    def process_data(self):
        # Extract Features
        features = self.data.drop(columns=['label'])

        # Extract Labels
        labels = self.data[['label']]

        # Change Labels to 1 or 0
        OHE_labels = labels.where(labels!="adenomas_and_adenocarcinomas", other=1)
        OHE_labels = OHE_labels.where(labels=="adenomas_and_adenocarcinomas", other=0)

        #Train Test SPlit
        X_Train, X_Test, Y_Train, Y_Test = sk.model_selection.train_test_split(features, OHE_labels, test_size=self.test_size)
        X_Train = X_Train.to_numpy(dtype=float)
        X_Test = X_Test.to_numpy(dtype=float)
        Y_Train = Y_Train.to_numpy(dtype=int)
        Y_Test = Y_Test.to_numpy(dtype=int)

        return X_Train, X_Test, Y_Train, Y_Test

    def train_classifier(self, X_Train, Y_Train):
        cls = sk.svm.SVC(gamma=self.gamma)
        cls.fit(X_Train, Y_Train)
        train_score = cls.score(X_Train, Y_Train)
        return cls, train_score

    def test_and_graph_results(self, cls, X_Test, Y_Test):
        X_predicted = cls.predict(X_Test)
        X_predicted_score = cls.score(X_Test, Y_Test)
        return X_predicted_score

if __name__ == "__main__":

    # Import Data and Functions
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

    # Classifier
    cfier = CancerClassifier(gamma=10, data=data_reduced)
    X_Train, X_Test, Y_Train, Y_Test = cfier.process_data()
    cls, train_score = cfier.train_classifier(X_Train, Y_Train.ravel())
    X_predicted_score = cfier.test_and_graph_results(cls, X_Test, Y_Test.ravel())
    print(X_predicted_score)






