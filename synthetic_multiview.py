from logging_setup import logger

from data_source import DataViewGenerator

import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import scipy
from scipy.io import loadmat

class AlteredDataSet(DataViewGenerator):
    #def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
    def __init__(self, *args, **kwargs):
        """
        Multiview dataset generator that generates altered views from true labels
        :param nVHigh: Number of views with a high amount of incorrect data point labels
        :param h: Percentage of incorrect data point labels considered as 'high'
        :param l: Percentage of incorrect data point labels considered as 'low'
        :param nC: Number of true clusters
        :param nP: Number of data points
        :param nV: Number of views
        :param seed: Seed to initialize the rnd nr generator.
        """
        self.logger = logger
        self.logger.debug("args: {0}".format(args))
        self.logger.debug("kwargs: {0}".format(kwargs))
        if "args" in kwargs:
            self.logger.debug("getting args from kwargs!")
            dict_args = kwargs["args"]
            logger.debug("params:{0}".format(dict_args))
            if "nVHigh" in dict_args:
                self.nVHigh = int(dict_args["nVHigh"])
            else:
                self.nVHigh = 1
            if "h" in dict_args:
                self.h = float(dict_args["h"])
            else:
                self.h = 0.3
            if "l" in dict_args:
                self.l = float(dict_args["l"])
            else:
                self.l = 0.1
            if "nC" in dict_args:
                self.nC = int(dict_args["nC"])
            else:
                self.nC = 10
            if "nP" in dict_args:
                self.nP = int(dict_args["nP"])
            else:
                self.nP = 100
            if "nV" in dict_args:
                self.nV = int(dict_args["nV"])
            else:
                self.nV = 5
            if "seed" in dict_args:
                self.seed = int(dict_args["seed"])
            else:
                self.seed = 11011
        elif kwargs != None:
            self.logger.error("Getting dataset generation parameters from args is not implemented!")
            raise NotImplemented

        self.logger.debug("starting with parameters {0}".format({'seed':self.seed, 'h':self.h, 'nVHigh':self.nVHigh,
                                                                 'l':self.l, 'nC':self.nC, 'nP':self.nP, 'nV':self.nV}))

        self.name = 'AltDS_{0}_{1:.2f}'.format(self.nVHigh, self.h)

        self.data_views = None
        self.views = {}
        self.__build_views__()

    def __build_views__(self):
        self.logger.debug("building the views!")
        # generate nP integers in [0, nC[ for the true labels
        trueLabels = np.random.randint(self.nC, size=(self.nP,))
        self.labels = trueLabels
        # construction of views
        _ = [self.views.setdefault('v{0}'.format(i), None) for i in range(self.nV)]
        #nViewsLow = int(np.ceil(self.nV * self.nVHigh))
        #nViewsHigh = self.nVHigh
        #nViewsLow = self.nV - nViewsHigh
        #highlyAlteredViews = set(np.random.choice(list(self.views.keys()), self.nVHigh, replace=False))
        highlyAlteredViews = sorted(list(self.views.keys()))[:self.nVHigh]
        for i in self.views:
            self.views[i] = trueLabels.copy()
            viewPtr = self.views[i]
            factor = self.h if i in highlyAlteredViews else self.l
            for ci in range(self.nC): # foreach cluster c
                cluster = np.where(viewPtr == ci)[0]
                nci = len(cluster)
                sampleSz = int(np.ceil(factor * nci)) # size of the sample
                #sampleSz = factor  # size of the sample
                sampledPts = np.random.choice(range(nci), sampleSz, replace=False) # sampled pts in the cluster
                sampledLabels = np.random.choice(list(set(range(self.nC)) - set([ci])), sampleSz)
                viewPtr[cluster[sampledPts]] = sampledLabels
                #viewPtr[cluster[sampledPts]] = self.nC + 1


if __name__ == '__main__':
    #adSet = AlteredDataSet(seed=23, pctHigh=0.23, h=.9, l=0.8, nC=10, nP=100, nV=10)
    #adSet = AlteredDataSet(args={'seed' : 23, 'nVHigh' : 0.5, 'h' : .4, 'l' : .1, 'nC' : 5, 'nP' : 200, 'nV' : 6})
    adSet = AlteredDataSet(args={'seed': 1, 'nVHigh': 4, 'h': 0.25, 'l': 0.05, 'nC': 5, 'nP': 200, 'nV': 6})
    logger.debug("Created dataset {0}".format(adSet.name))
    for v in adSet.get_views():
        logger.debug('View "{0}", P: {1:.3f}, F1: {2:.3f}, MI: {3:.3f}, ARI: {4:.3f}'.format(v,
                                              metrics.precision_score(adSet.get_real_labels(), adSet.get_views()[v], average='weighted'),
                                            metrics.f1_score(adSet.get_real_labels(), adSet.get_views()[v], average='weighted'),
                                            metrics.normalized_mutual_info_score(adSet.get_real_labels(), adSet.get_views()[v]),
                                            metrics.adjusted_rand_score(adSet.get_real_labels(), adSet.get_views()[v])
                                                          )
                     )


