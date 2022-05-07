import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def pca_reduce(
    data: pd.DataFrame, 
    n_components: int = 100,
    label_col_name: str = 'label',
    remove_sample_col = True
) -> pd.DataFrame:
    """
    # PCA Reduce
    This function will take a pandas dataframe object of our feature data and
    run dimensionality reduction on it. It first removes the sample id column
    then runs the calculation if you've specified this. This assumes that the 
    sample id column is the first column in the dataframe.
    """
    # extract labels to append after dimensionality rection
    _labels = data[label_col_name]
    del data[label_col_name]

    # drop the samples column
    if remove_sample_col:
        data_no_samples = data.iloc[:,1:]
    else:
        data_no_samples = data

    # init PCA object and fit
    pca = PCA(n_components=n_components)
    SS = StandardScaler(copy=True, with_mean=True, with_std=True)
    SS.fit(data_no_samples)
    data_scaled = SS.transform(data_no_samples)

    pca.fit(data_scaled)

    # log out the results
    print(
        f"{round(sum(pca.explained_variance_ratio_)*100,2)}% variance explained with first {n_components} components."
    )

    # run the reduction
    dim_reduced = pca.transform(data_scaled)

    # rebuild a dataframe
    cols = [f"PCA{i+1}" for i in range(n_components)]
    df_reduced = pd.DataFrame(columns=cols, data=dim_reduced)

    # add labels back
    df_reduced[label_col_name] = _labels

    return [df_reduced, pca.explained_variance_ratio_]

def plot_pca(
    data: pd.DataFrame, 
    x: str = "PCA1", 
    y: str = "PCA2",
    label_col: str = "label"
) -> None:
    """
    # Plot PCA
    Plot the results of the PCA from the PCA_reduce function. This
    function assumes that you've passed a dataframe with columns
    named 'PCA1' and 'PCA2'. You can change this if you want
    """
    _, ax = plt.subplots(1,1)
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        hue=label_col
    )
    plt.show()

def tSNE_reduce(
    data: pd.DataFrame, 
    n_components: int = 2,
    label_col_name: str = 'label',
    remove_sample_col = True
) -> pd.DataFrame:
    """
    # tSNE Reduce
    This is similar to the PCA reduction function, however, this one uses
    a more advanced tSNE approach. It usually takes longer to do this, but
    can often yield better results. 
    
    This again assumes that the sample id column is the first column in
    the dataframe.
    """
    _labels = data[label_col_name]
    del data[label_col_name]

    # drop the samples column
    if remove_sample_col:
        data_no_samples = data.iloc[:,1:]
    else:
        data_no_samples = data
    
    # fit data to tSNE
    tSNE_data = TSNE(
        n_components=n_components,
        learning_rate='auto',
        init='random'
    ).fit_transform(data_no_samples)

    # build columns back and regenerate data frame
    cols = [f"tSNE{i+1}" for i in range(n_components)]
    data_reduced = pd.DataFrame(
        data=tSNE_data,
        columns=cols
    )
    data_reduced[label_col_name] = _labels

    return data_reduced

def pca_bar(
    data: np.ndarray,
    n_components: int = None
) -> None:
    # they didnt specify
    if n_components is None:
        print(f"Creating bar with all components (n={len(data)})")
        N = len(data)
    # they specified, but it was too many
    elif n_components > len(data):
        print(f"Request number of components ({n_components}) is greater than length of array provided ({len(data)})")
        print(f"Using entire array instead... ")
        N = len(data)
    # they got it right :)
    else:
        N = n_components

    # tranform to percentages
    data = np.round(data,4)*100

    _labels = [f"PC{i+1}" for i in range(N)]

    _, ax = plt.subplots(1,1)
    sns.barplot(
        y=data,
        x=_labels,
        color="b",
        ax=ax
    )

    ax.set_title(f"Explained variance in first {N} Principle Components\nTotal = {sum(data)}%")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explain Variance (%)")
    ax.bar_label(ax.containers[0])
    plt.show()

def plot_tSNE(
    data: pd.DataFrame,
    x: str = "tSNE1",
    y: str = "tSNE2",
    label_col: str = 'label'
) -> None:
    """
    # Plot tSNE
    This function is similar to the plot PCA function and
    is intended to plot the results of the tSNE_reduce
    function.
    """
    _, ax = plt.subplots(1,1)
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        hue=label_col
    )
    plt.show()


#
# TEST CODE
#
if __name__ == '__main__':
    # read in feature data and rename
    feature_data = pd.read_csv("data/demo_feature_file.csv")
    feature_data.rename(columns={'samples': 'id'}, inplace=True)

    # read in labels
    labels = pd.read_csv("data/demo.csv")

    # join on id
    data_labeled = feature_data.merge(labels, on="id", how="inner")

    data_reduced, explained_variance = pca_reduce(data_labeled, n_components=10)
    plot_pca(data_reduced)
    pca_bar(
        explained_variance,
        n_components=10
    )

    # # run tSNE
    # feature_data = pd.read_csv("data/demo_feature_file.csv")
    # feature_data.rename(columns={'samples': 'id'}, inplace=True)

    # # read in labels
    # labels = pd.read_csv("data/demo.csv")

    # # join on id
    # data_labeled = feature_data.merge(labels, on="id", how="inner")

    # data_reduced = tSNE_reduce(data_labeled, n_components=2)
    # plot_tSNE(data_reduced)