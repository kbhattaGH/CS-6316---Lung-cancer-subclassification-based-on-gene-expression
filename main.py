##
import numpy as np
import sklearn as sk


class CancerClassifier:
    def __init__(self, data, gamma, split):
        self.data = data
        self.Train = split[0]
        self.Test = split[1]
        self.gamma = gamma

    def process_data(self):
        #import data from processed file and extract required meta-data and train-tes split
        X_Train = self.data[0:int(self.Train/100*len(self.data)), 0]
        Y_Train = self.data[0:int(self.Train/100*len(self.data)), 1]
        X_Test = self.data[len(X_Train):len(self.data), 0]
        Y_Test = self.data[len(Y_Train):len(self.data), 1]
        return X_Train, Y_Train, X_Test, Y_Test

    def train_classifier(self, X_Train, Y_Train):
        cls = sk.svm.SVC(gamma=self.gamma)
        cls.fit(X_Train, Y_Train)
        train_score = cls.score(X_Train, Y_Train)
        return cls, train_score

    def test_and_graph_results(self, cls, X_Test, Y_Test):
        X_predicted = cls.predict(X_Test)
        X_predicted_score = cls.score(X_predicted, Y_Test)


#if __name__ == "__main__":
    #filename = ""
    #with open(filename,'r'):
        #f1= ...
        #cfier = CancerClassifier(f1, gamma=50, split=[80,20])
        #data = cfier.process_data()



