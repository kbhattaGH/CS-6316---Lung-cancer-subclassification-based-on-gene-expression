from sys import argv, stderr
from getopt import getopt, GetoptError
import pandas as pd


def assign_labels(classes):

    column_names = ["id", "label"]
    true_label = pd.DataFrame(columns=column_names)

    for c in classes:
        file_path = "../metadata/" + c + "_mfile.txt"
        df = pd.read_csv(file_path, usecols=["id"], sep="\t")
        df["label"] = c
        true_label = true_label.append(df, ignore_index=True)

    return true_label


def train_test_split(df, trainratio):

    # Creating the train with given train_ratio
    train = df.sample(frac=trainratio)
    train.to_csv("../data/train_labels.csv", index=False)
    # Creating test with the rest
    test = df.drop(train.index)
    test.to_csv("../data/test_labels.csv", index=False)


def main(class1, class2, trainratio):

    classes = [class1, class2]

    true_label = assign_labels(classes)
    train_test_split(true_label, trainratio)


if __name__ == "__main__":
    try:
        opts, args = getopt(
            argv[1:],
            "hm:x:g:e:",
            ["help", "class1=", "class2=", "trainratio="],
        )
    except GetoptError as err:
        print(err)
        print(__doc__, file=stderr)
        exit(1)

    class1 = "squamous_cell_neoplasms"
    class2 = "adenomas_and_adenocarcinomas"
    trainratio = 0.8

    for o, a in opts:
        if o in ("-h", "--help"):
            print(__doc__, file=stderr)
            exit()
        elif o in ("-c1", "--class1"):
            class1 = str(a)
        elif o in ("-c1", "--class2"):
            class2 = str(a)
        elif o in ("-r", "--trainratio"):
            trainratio = float(a)
        else:
            assert False, "unhandled option"

    main(class1, class2, trainratio)
