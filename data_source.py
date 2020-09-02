import logging
from logging_setup import logger

import numpy as np
import scipy
import csv
import os
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy.io import loadmat

from abc import ABC, abstractmethod, ABCMeta

from tabulate import tabulate
import sklearn.metrics as evalfns


def assess_datasource_views(D):
    header_list = ('View', 'F1 (macro)','Pr (macro)','Rec (macro)', 'NMI', 'Vm')
    assess_views = lambda C : [(vw, evalfns.f1_score(C.get_real_labels(), lbls, average='micro'),
                                evalfns.precision_score(C.get_real_labels(), lbls, average='micro'),
                                evalfns.recall_score(C.get_real_labels(), lbls, average='micro'),
                                evalfns.normalized_mutual_info_score(C.get_real_labels(), lbls),
                                evalfns.v_measure_score(C.get_real_labels(), lbls)) for vw, lbls in C.get_views().items()]

    print(tabulate(assess_views(D), headers=header_list, tablefmt='github', stralign='center', floatfmt='.3f', numalign='center'))


class DataViewGenerator(metaclass=ABCMeta):
    """
    Represents a set of views for a dataset.
    Each view is generated from the result of a clustering method over each data representation.
    """
    def __init__(self, views: dict, dataset_dir: object, NCLUSTERS: object, seed: object) -> object:
        """
        Initializes the View generator and creates the views by performing clustering.
        :param views: dict with view names associated to the file name that hast the data representation. This dict must contain a 'labels' and 'dataset' keys.
        :param dataset_dir: data path where the different data representations are stored.
        :param NCLUSTERS: int when all views have the same nr. of clusters or a dict (tfidf,skipgram,lda) that allows to assign each view a different cluster number.
        :param seed: random generator seed.
        """
        # self.logger = logging.getLogger(__name__)
        self.logger = logger
        #self.name = self.__class__.__name__  # can be overrided if neccessary
        self.data_views = views
        self.labels_file = views["labels"]
        self.name = views["dataset"]
        del self.data_views['labels']
        del self.data_views['dataset']
        self.dataset_dir = dataset_dir
        if(isinstance(NCLUSTERS, int)):
            #NCLUSTERS = {'tfidf':NCLUSTERS, 'lda':NCLUSTERS, 'skipgram':NCLUSTERS}
            self.NCLUSTERS = {}
            for v in self.data_views:
                #if not v in ["labels","dataset"]:
                self.NCLUSTERS[v] = NCLUSTERS
        elif isinstance(NCLUSTERS, dict):
            self.NCLUSTERS = NCLUSTERS
        else:
            raise NotImplemented

        self.seed = seed
        self.views = {}
        self.labels = None

        self.__build_views__()

    def __build_views__(self):
        np.random.seed(self.seed)
        #random_state = np.random.randint(2 ** 16 - 1)
        random_state = self.seed
        with open(self.dataset_dir + os.sep + self.labels_file.replace("?", os.sep), 'r') as f:
            reader = csv.reader(f)
            lst_labels = list(reader)

        lst_labels = [x[0] for x in lst_labels]
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(lst_labels)

        #for viewname in ["tfidf", "lda", "skipgram"]:
        for viewname in self.data_views:
            #if viewname in ["labels","dataset"]:
            #    continue

            X = None
            #if viewname == 'tfidf': # uses a sparse representation, thus it must be loaded differently
            #    X = scipy.sparse.load_npz(self.dataset_dir + os.sep + self.data_views[viewname].replace("?", os.sep))
            #else:
            #    X = np.load(self.dataset_dir + os.sep + self.data_views[viewname].replace("?", os.sep))['arr_0']

            data_repr_input = self.dataset_dir + os.sep + self.data_views[viewname].replace("?", os.sep)
            try:
                # trying to open the data representation as a dense file
                X = np.load(data_repr_input)['arr_0']
            except KeyError as e:
                #logger.debug(e.__str__())
                logger.debug("Opening data representation for view %s in sparse format" % (viewname))
                X = scipy.sparse.load_npz(data_repr_input)

            km = MiniBatchKMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
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

    def create_view(self, new_k):
        np.random.seed(self.seed)
        # random_state = np.random.randint(2 ** 16 - 1)
        random_state = self.seed

        for viewname in self.data_views:
            X = self.data_views[viewname]
            # km = KMeans(n_clusters=new_k, init='k-means++', random_state=random_state)
            km = MiniBatchKMeans(n_clusters=new_k, init='k-means++', random_state=random_state)
            km_labels = km.fit_predict(X)
            self.views["{0}_K{1}".format(viewname, new_k)] = km_labels




######################## Below ...Old datasets useed in the HII conference!
"""
TwentyNewsgroupView
"""
class TwentyNewsgroupView(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        data_views = {"lda": "20Newsgroup?20ng_4groups_lda.npz",
                      "skipgram": "20Newsgroup?20ng_4groups_doc2vec.npz",
                      "tfidf": "20Newsgroup?20ng-scikit-mat-tfidf.npz",
                      "labels": "20Newsgroup?20ng_4groups_labels.csv",
                      "dataset": "20Newsgroup"}
        #super(TwentyNewsgroupView, self).__init__(dataset_dir, NCLUSTERS, seed)
        DataViewGenerator.__init__(self, data_views, dataset_dir, NCLUSTERS, seed)


class BBCSportsView(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        data_views = {"lda": "bbcsport?fulltext?bbcsport-textdata-mat-lda.npz",
                      "skipgram": "bbcsport?fulltext?bbcsport-textdata-mat-skipgram.npz",
                      "tfidf": "bbcsport?fulltext?bbcsport-textdata-mat-tfidf.npz",
                      "labels": "bbcsport?fulltext?bbcsport-textdata-labels.csv",
                      "dataset": "BBCSport"}
        #super(BBCSportsView, self).__init__(dataset_dir, NCLUSTERS, seed)
        DataViewGenerator.__init__(self, data_views, dataset_dir, NCLUSTERS, seed)


class ReutersView(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        data_views = {"lda": "reuters-r8?r8-test-mat-lda.npz",
                      "skipgram": "reuters-r8?r8-test-mat-skipgram.npz",
                      "tfidf": "reuters-r8?r8-test-mat-tfidf.npz",
                      "labels": "reuters-r8?r8-test-labels.txt",
                      "dataset": "Reuters-R8"}
        #super(ReutersView, self).__init__(dataset_dir, NCLUSTERS, seed)
        DataViewGenerator.__init__(self, data_views, dataset_dir, NCLUSTERS, seed)


class WEBKBView(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        data_views = {"lda": "WebKB?webkb-textdata-mat-lda.npz",
                      "skipgram": "WebKB?webkb-textdata-mat-skipgram.npz",
                      "tfidf": "WebKB?webkb-textdata-mat-tfidf.npz",
                      "labels": "WebKB?webkb-textdata-labels.csv",
                      "dataset": "WebKB"}
        #super(WEBKBView, self).__init__(dataset_dir, NCLUSTERS, seed)
        DataViewGenerator.__init__(self, data_views, dataset_dir, NCLUSTERS, seed)

