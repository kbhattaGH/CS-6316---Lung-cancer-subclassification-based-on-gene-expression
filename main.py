import sklearn as sk
import pandas as pd
import sys
import os
import matplotlib.pyplot as plot
import numpy as np

class CancerClassifier:
    def __init__(self, parameters, data, test_size=0.33, search_best_val=True):
        self.search_best_val = search_best_val
        self.parameters = parameters
        self.gamma = self.parameters["gamma"][0]
        self.data = data
        self.test_size = test_size
        self.C = self.parameters["C"][0]
        self.kernel = self.parameters["kernel"][0]

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

    def train_classifier(self, X_Train, Y_Train, X_Test, Y_Test):

        if self.search_best_val == True:
            print("Searching for the best hyper-parameters")
            cls_init = sk.svm.SVC()


            cls = sk.model_selection.GridSearchCV(cls_init, self.parameters, verbose=True, n_jobs=-1)

            cls.fit(np.concatenate((X_Train, X_Test), axis=0), np.concatenate((Y_Train, Y_Test), axis=0).ravel())

            self.kernel, self.C, self.gamma = cls.best_params_["kernel"], cls.best_params_["C"], cls.best_params_["gamma"]
            print("The best parameters for the problem are: \n", pd.DataFrame(cls.best_params_, index=["Parameters"]))

        cls = sk.svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        cls.fit(X_Train, Y_Train.ravel())
        train_score = cls.score(X_Train, Y_Train.ravel())
        return cls, train_score

    def test_and_graph_results(self, cls, X_Test, Y_Test):
        Y_predicted = cls.predict(X_Test)
        Y_predicted_score = cls.score(X_Test, Y_Test.ravel())
        scatter = plot.scatter(X_Test[:,1], X_Test[:,2], c=Y_predicted)
        plot.xlabel("PCA1")
        plot.ylabel("PCA2")
        plot.legend(handles=scatter.legend_elements()[0])
        plot.show()
        return Y_predicted_score, Y_predicted

if __name__ == "__main__":
    print("Begin.")
    # Import Data and Functions
    abspath = sys.path[0]
    src_path = os.path.abspath(os.path.join(abspath, 'src'))
    sys.path.append(src_path)
    from dimensionality_reduction import pca_reduce, plot_pca, pca_bar

    print("Reading data.")
    data_path = os.path.join(abspath, 'data')
    feature_data1 = pd.read_csv(data_path+'/full/train_features.csv')
    feature_data2 = pd.read_csv(data_path+'/full/test_features.csv')
    data_labeled = pd.concat([feature_data1, feature_data2], ignore_index=True)
    data_labeled.rename(columns={'samples': 'id'}, inplace=True)


    # Reduced data
    print("Running dimensionality reduction.")
    data_reduced_pca, explained_variance = pca_reduce(data_labeled, n_components=1200)

    plot_pca(data_reduced_pca)
    pca_bar(
        explained_variance,
        n_components=100,
        print_labels=False
    )
    print("PCA done.")


    # Classifier hyperparameter tuning
    print("Building classifier.")
    params = {"kernel": ["linear", "rbf"], "C": [0.1, 0.5, 1, 1.5], "gamma": [10 ** -5, 10 ** -4, 10 ** -2]}
    cfier = CancerClassifier(parameters=params, data=data_reduced_pca)
    X_Train, X_Test, Y_Train, Y_Test = cfier.process_data()

    print("Training.")
    cls, train_score = cfier.train_classifier(X_Train, Y_Train.ravel(), X_Test, Y_Test.ravel())

    print("Testing.")
    Y_predicted_score,  Y_predicted = cfier.test_and_graph_results(cls, X_Test, Y_Test.ravel())

    print(f"Training accuracy: {round(train_score*100,2)}%")
    # print(Y_predicted)
    print(f"Testing accuracy: {round(Y_predicted_score*100,2)}%")






