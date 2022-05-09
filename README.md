# CS6316 Lung Cancer Sub-Classification Using the Tumor Transcriptome

## Downloading Data from GDC Data Portal Using Manifest Files

1. create new directories in the `data` directory using the classifiers as directory name
```
cd data
mkdir squamous_cell_neoplasms adenomas_and_adenocarcinomas
```
2. downloading all transcriptome profiling files for *squamous cell neoplasms* and *adenomas and adenocarcinomas* lung cancer
```
cd squamous_cell_neoplasms
gdc-client download -m ../../metadata/squamous_cell_neoplasms_mfile.txt

cd ../adenomas_and_adenocarcinomas
gdc-client download -m ../../metadata/adenomas_and_adenocarcinomas_mfile.txt

```
## Generat and Split True Lable Table into Train and Test Sets
```
python3 extract_true_label.py squamous_cell_neoplasms adenomas_and_adenocarcinomas
```
## Generat Feature Matrix for Train and Test Sets
The feature matrix contains samples as rows and features (genes) as column. The FPKM score from the input transcriptome profiling files are used as the feature score
```
python3 generate_featrue_matrix.py train

python3 generate_featrue_matrix.py test
```

##Running Program

1. Move pre-processed data into the "data/full" directory.
2. pca_reduce function arguments required: 
```
**n_components** Number of PCA components
**Data** Data to reduce

```
4. CancerClassifier Method Arguments required: 

```
**parameters** Dictionary of hyper parameters for tuning

**data** Data with sample id and gene expression scores

**test_size** Test:Train split ratio

**search_best_val** boolean for turning on hyperparameter tuning. If false uses first value of each hyperparameter in parameter dict.

```
