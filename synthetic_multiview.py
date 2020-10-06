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
        self.logger = logger
        self.logger.debug("args: {0}".format(args))
        self.logger.debug("kwargs: {0}".format(kwargs))
        if "args" in kwargs:
            self.logger.debug("getting args from kwargs!")
            dict_args = kwargs["args"]
            logger.debug("params:{0}".format(dict_args))
            if "pctLow" in dict_args:
                self.pctLow = float(dict_args["pctLow"])
            else:
                self.pctLow = 0.1
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

        self.logger.debug("starting with parameters {0}".format({'seed':self.seed, 'h':self.h, 'pctLow':self.pctLow,
                                                                 'l':self.l, 'nC':self.nC, 'nP':self.nP, 'nV':self.nV}))
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
        nViewsLow = int(np.ceil(self.nV * self.pctLow))
        nViewsHigh = self.nV - nViewsLow
        highlyAlteredViews = set(np.random.randint(self.nV, size=(nViewsHigh,))) # randomly pick
        for i in self.views:
            self.views[i] = trueLabels.copy()
            viewPtr = self.views[i]
            factor = self.h if i in highlyAlteredViews else self.l
            for ci in range(self.nC): # foreach cluster c
                cluster = np.where(viewPtr == ci)[0]
                nci = len(cluster)
                sampleSz = int(np.round(factor * nci)) # size of the sample
                sampledPts = np.random.choice(range(nci), sampleSz) # sampled pts in the cluster
                viewPtr[cluster[sampledPts]] = self.nC + 1


if __name__ == '__main__':
    #adSet = AlteredDataSet(seed=23, pctHigh=0.23, h=.9, l=0.8, nC=10, nP=100, nV=10)
    adSet = AlteredDataSet(args={'seed' : 23, 'pctLow' : 0.25, 'h' : .9, 'l' : 0.6, 'nC' : 5, 'nP' : 50, 'nV' : 6})
    for v in adSet.get_views():
        logger.debug('View "{0}", AvgP: {1}'.format(v,
                                              metrics.precision_score(adSet.get_real_labels(), adSet.get_views()[v], average='micro')))


