##
import numpy as np
import sklearn as sk

class CancerClassifier:
    def __init__(self,'filename','gamma'):
        self.feature_size = 0;
        self.features = None;
        self.labels = None
        self.classifier = None

    def process_data(self):
    #import data from processed file and extract required meta-data and train-tes split
        return self.feature_size, self.features, self.labels

    def train_classifier(self):
    #Train svm
        return self.classifier

    def test_classifier(self):
        #test svm on test set

    def graph_results:
    #as the name suggests

if __name__ == "__main__":
    filename = ""
    with open (filename):
        cfier = CancerClassifier('filename','gamma')
        data=cfier.process_data()



