# Multiview-clustering

## Anaconda environment

The file `environment.yml` contains the list of packages required to reproduce the experiments. To create the same environment, execute:
```bash
$ conda env create -f environment.yml
```


## Running the experiments

Be aware of adding the data directory parameter in [`line 716`](https://github.com/jfzo/Multiview-clustering/blob/bd4ff54f4e7642b6ee62d4d3a3ca3a60a129e804/NCF_clustering_fusion.py#L716) of file NCF_clustering_fusion.py.
```
$ python.exe experiments_parallel.py
```
