from sys import argv, stderr
from getopt import getopt, GetoptError
import os
import pandas as pd


def extract_features(file):
    id = file.split("/")[-2]
    # print(id)
    df = pd.read_csv(file, skiprows=6, header=None, sep="\t")
    new_row = {"samples": [id]}
    # print(df)
    for index, row in df.iterrows():
        new_row[row[0]] = [row[7]]
    return pd.DataFrame.from_dict(new_row)


def main(type):
    if type == "train":
        meta = "../data/train_labels.csv"
    elif type == "test":
        meta = "../data/train_labels.csv"

    list = pd.read_csv(meta, usecols=["id"])
    list = list["id"].tolist()

    rootdir = "../data"
    count = 0

    features = pd.DataFrame(columns=["samples"])

    for subdir, dirs, files in os.walk(rootdir):
        if any(ele in subdir for ele in list) and "logs" not in subdir:
            for file in files:
                if "annotations" not in file:
                    f = os.path.join(subdir, file)
                    features = pd.concat(
                        [features, extract_features(f)], axis=0, ignore_index=True
                    )
                    print(features)
                    count += 1
    features.to_csv(os.path.join("..", "data", type + "_features.csv"), index=False)


if __name__ == "__main__":
    try:
        opts, args = getopt(
            argv[1:],
            "hm:x:g:e:",
            ["help", "type="],
        )
    except GetoptError as err:
        print(err)
        print(__doc__, file=stderr)
        exit(1)

    type = "train"

    for o, a in opts:
        if o in ("-h", "--help"):
            print(__doc__, file=stderr)
            exit()
        elif o in ("-t", "--type"):
            class1 = str(a)
        else:
            assert False, "unhandled option"

    main(type)
