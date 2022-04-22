#!/usr/bin/env python

"""
    usage:
        extract_true_label [options] type
    where the options are:
        -h,--help : print usage and quit
"""

from sys import argv, stderr
from getopt import getopt, GetoptError
import os
import pandas as pd


def extract_features(file):
    """
    extract features of given transcriptome profiling file
    each transcriptome profiling file result as row in the feature matrix

    param
    --------
        file: transcriptome profiling file path

    return
    --------
        a single row dataframe containg sample name, feature, and scores
    """

    # get the sampel id
    id = file.split("/")[-2]
    # load dataframe
    df = pd.read_csv(file, skiprows=6, header=None, sep="\t")
    # create dict for the data
    new_row = {"samples": [id]}
    # iterate through the df
    for index, row in df.iterrows():
        # add feature (gene) and score (fpkm) to the dict
        new_row[row[0]] = [row[7]]

    # convert dict to single row dataframe
    return pd.DataFrame.from_dict(new_row)


def main(type):
    """
    generate feature matrix:
        row: samples
        col: feature score (fpkm score)
    saved as csv file

    param
    -------
        type: train or test
    """

    # get the ture label file of the given type (train/test)
    # contains the samples of that set
    if type == "train":
        meta = "../data/train_labels.csv"
    elif type == "test":
        meta = "../data/test_labels.csv"
    # get the list of samples of train or test set
    list = pd.read_csv(meta, usecols=["id"])
    list = list["id"].tolist()

    rootdir = "../data"
    count = 0

    # init feature matrix
    features = pd.DataFrame(columns=["samples"])
    # iterate dir containing sample data
    for subdir, dirs, files in os.walk(rootdir):
        # only select files in the correct set
        # skip the logs dir
        if any(ele in subdir for ele in list) and "logs" not in subdir:
            for file in files:
                # exclude annotation files
                if "annotations" not in file:
                    f = os.path.join(subdir, file)
                    # extract featrues from the transcriptome profiling file
                    # concat to the feature matrix
                    features = pd.concat(
                        [features, extract_features(f)], axis=0, ignore_index=True
                    )
                    count += 1
                    print(count)
    features = features.fillna(0)
    features.to_csv(os.path.join("..", "data", type + "_features.csv"), index=False)


if __name__ == "__main__":
    try:
        opts, args = getopt(
            argv[1:],
            "hm:x:g:e:",
            ["help"],
        )
    except GetoptError as err:
        print(err)
        print(__doc__, file=stderr)
        exit(1)

    for o, a in opts:
        if o in ("-h", "--help"):
            print(__doc__, file=stderr)
            exit()
        else:
            assert False, "unhandled option"

    main(args[0])
