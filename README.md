# Multiview-clustering

## Anaconda environment

The file `environment.yml` contains the list of packages required to reproduce the experiments. To create the same environment, execute:
```bash
$ conda env create -f environment.yml
```

## Creating the views for each dataset

```
$ python3 3-view-processing.py --i ./data/reuters-r8/r8-test.csv --o ./data/reuters-r8/r8-test-mat-skipgram.npz --type skipgram
$ python3 3-view-processing.py --i ./data/reuters-r8/r8-test.csv --o ./data/reuters-r8/r8-test-mat-tfidf.npz --type tfidf
$ python3 3-view-processing.py --i ./data/reuters-r8/r8-test.csv --numtopics 8 --o ./data/reuters-r8/r8-test-mat.npz --type lda
```

## Running the experiments

Be aware of adding the data directory parameter in `line 716` of file NCF_clustering_fusion.py.
```
$ python3 NCF_clustering_fusion.py
```
