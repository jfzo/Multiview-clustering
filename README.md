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

Be aware of adding the data directory parameter in [`line 716`](https://github.com/jfzo/Multiview-clustering/blob/bd4ff54f4e7642b6ee62d4d3a3ca3a60a129e804/NCF_clustering_fusion.py#L716) of file NCF_clustering_fusion.py.
```
$ python.exe NCF_clustering_fusion.py --nclusters 3 5 10 15 --nruns 5 --logfile ga_ncf_run.log
```
