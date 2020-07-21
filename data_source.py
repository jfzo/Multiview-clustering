import logging
from logging_setup import logger

import numpy as np
import scipy
import csv
import os
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans, KMeans

from abc import ABC, abstractmethod


class DataViewGenerator(ABC):
    """
    Represents a set of views for a dataset.
    Each view is generated from the result of a clustering method over each data representation.
    """

    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        """
        Initializes the View generator.
        :param dataset_dir: data path where the different data representations are stored.
        :param NCLUSTERS: int when all views have the same nr. of clusters or a dict (tfidf,skipgram,lda) that allows to assign each view a different cluster number.
        :param seed: random generator seed.
        """
        # self.logger = logging.getLogger(__name__)
        self.logger = logger
        self.name = self.__class__.__name__  # can be overrided if neccessary
        self.dataset_dir = dataset_dir
        if(isinstance(NCLUSTERS, int)):
            NCLUSTERS = {'tfidf':NCLUSTERS, 'lda':NCLUSTERS, 'skipgram':NCLUSTERS}
        self.NCLUSTERS = NCLUSTERS

        self.seed = seed
        self.views = {}
        self.labels = None

        self.__build_views__()

    def __build_views__(self):
        np.random.seed(self.seed)
        random_state = np.random.randint(2 ** 16 - 1)

        with open(self.dataset_dir + os.sep + self.data_views["labels"].replace("?", os.sep), 'r') as f:
            reader = csv.reader(f)
            lst_labels = list(reader)

        lst_labels = [x[0] for x in lst_labels]
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(lst_labels)

        for viewname in ["tfidf", "lda", "skipgram"]:
            X = None
            if viewname == 'tfidf':
                X = scipy.sparse.load_npz(self.dataset_dir + os.sep + self.data_views[viewname].replace("?", os.sep))
            else:
                X = np.load(self.dataset_dir + os.sep + self.data_views[viewname].replace("?", os.sep))['arr_0']
            km = MiniBatchKMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', n_init=3, init_size=900, batch_size=300,
                                 random_state=random_state)

            km_labels = km.fit_predict(X)
            self.views[viewname] = km_labels

    def get_views(self):
        if self.labels is not None:
            return self.views
        else:
            logger.error("Views not yet extracted!")
            return None

    def get_real_labels(self):
        if self.labels is None:
            logger.error("Views not yet extracted!")
        return self.labels

    #@abstractmethod
    #def __build_views__(self):
    #    pass


class TwentyNewsgroupView(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.data_views = {"lda": "20Newsgroup?20ng_4groups_lda.npz",
                      "skipgram": "20Newsgroup?20ng_4groups_doc2vec.npz",
                      "tfidf": "20Newsgroup?20ng-scikit-mat-tfidf.npz",
                      "labels": "20Newsgroup?20ng_4groups_labels.csv",
                      "dataset": "20Newsgroup"}
        super(TwentyNewsgroupView, self).__init__(dataset_dir, NCLUSTERS, seed)


class BBCSportsView(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.data_views = {"lda": "bbcsport?fulltext?bbcsport-textdata-mat-lda.npz",
                      "skipgram": "bbcsport?fulltext?bbcsport-textdata-mat-skipgram.npz",
                      "tfidf": "bbcsport?fulltext?bbcsport-textdata-mat-tfidf.npz",
                      "labels": "bbcsport?fulltext?bbcsport-textdata-labels.csv",
                      "dataset": "BBCSport"}
        super(BBCSportsView, self).__init__(dataset_dir, NCLUSTERS, seed)


class ReutersView(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.data_views = {"lda": "reuters-r8?r8-test-mat-lda.npz",
                      "skipgram": "reuters-r8?r8-test-mat-skipgram.npz",
                      "tfidf": "reuters-r8?r8-test-mat-tfidf.npz",
                      "labels": "reuters-r8?r8-test-labels.txt",
                      "dataset": "Reuters-R8"}
        super(ReutersView, self).__init__(dataset_dir, NCLUSTERS, seed)


class WEBKBView(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.data_views = {"lda": "WebKB?webkb-textdata-mat-lda.npz",
                      "skipgram": "WebKB?webkb-textdata-mat-skipgram.npz",
                      "tfidf": "WebKB?webkb-textdata-mat-tfidf.npz",
                      "labels": "WebKB?webkb-textdata-labels.csv",
                      "dataset": "WebKB"}
        super(WEBKBView, self).__init__(dataset_dir, NCLUSTERS, seed)
