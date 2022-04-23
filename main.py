import sklearn as sk
import pandas as pd
import sys
import os
import matplotlib.pyplot as plot


class CancerClassifier:
    def __init__(self, gamma, C, data, test_size=0.33):
        self.gamma = gamma
        self.data = data
        self.test_size = test_size
        self.C = C

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
        cls = sk.svm.SVC(C=self.C, gamma=self.gamma)
        cls.fit(X_Train, Y_Train.ravel())
        train_score = cls.score(X_Train, Y_Train.ravel())
        return cls, train_score

    def test_and_graph_results(self, cls, X_Test, Y_Test):
        Y_predicted = cls.predict(X_Test)
        Y_predicted_score = cls.score(X_Test, Y_Test.ravel())
        plot.scatter(X_Test[:,1], X_Test[:,2], c=Y_predicted)
        plot.show()
        return Y_predicted_score, Y_predicted

if __name__ == "__main__":

    # Import Data and Functions
    abspath = sys.path[0]
    src_path = os.path.abspath(os.path.join(abspath, 'src'))
    sys.path.append(src_path)
    from dimensionality_reduction import pca_reduce
    data_path = os.path.join(abspath, 'data')
    feature_data1 = pd.read_csv(data_path+'/train_features.csv')
    feature_data2 = pd.read_csv(data_path+'/test_features.csv')
    data_labeled = pd.concat([feature_data1, feature_data2], ignore_index=True)
    data_labeled.rename(columns={'samples': 'id'}, inplace=True)


    # Reduced data
    data_reduced = pca_reduce(data_labeled, n_components=1200)
    print(data_reduced)

    # Classifier
    cfier = CancerClassifier(gamma=10**-5, C=1, data=data_reduced)
    X_Train, X_Test, Y_Train, Y_Test = cfier.process_data()
    cls, train_score = cfier.train_classifier(X_Train, Y_Train.ravel())
    Y_predicted_score,  Y_predicted = cfier.test_and_graph_results(cls, X_Test, Y_Test.ravel())
    print(train_score)
    print(Y_predicted)
    print(Y_predicted_score)






