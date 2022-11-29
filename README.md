# Multiview-clustering

## Anaconda environment

The file `environment.yml` contains the list of packages required to reproduce the experiments. To create the same environment, execute:
```bash
$ conda env create -f environment.yml
```


## Running the experiments

Be aware of adding the data directory parameter in [`line 716`](https://github.com/jfzo/Multiview-clustering/blob/72f9a28c8c80ad4c40eb8792e3910b7476d04bce/experiments_parallel.py#L187) of file experiments_parallel.py. See also the other experimental settings in the same fiel.

```
$ python.exe experiments_parallel.py
```
