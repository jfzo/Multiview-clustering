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

    print(tabulate(assess_views(D), headers=header_list, tablefmt='fancy_grid', stralign='center', floatfmt='.3f', numalign='center'))


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



"""
Handwritten
"""
class Handwritten(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.logger = logger
        self.seed = seed
        mlabData = loadmat(dataset_dir)
        # self.name = self.__class__.__name__  # can be overrided if neccessary
        self.data_views = {"pix":mlabData['X'][0,0],
                      "fou":mlabData['X'][0,1],
                      "fac":mlabData['X'][0,2],
                      "zer":mlabData['X'][0,3],
                      "kar": mlabData['X'][0, 4],
                      "mor":mlabData['X'][0,5]}
        self.views = {}
        self.labels = [lbl[0] for lbl in mlabData['Y']]
        self.name = 'Handwritten'
        if (isinstance(NCLUSTERS, int)):
            # NCLUSTERS = {'tfidf':NCLUSTERS, 'lda':NCLUSTERS, 'skipgram':NCLUSTERS}
            self.NCLUSTERS = {}
            for v in self.data_views:
                # if not v in ["labels","dataset"]:
                self.NCLUSTERS[v] = NCLUSTERS
        elif isinstance(NCLUSTERS, dict):
            self.NCLUSTERS = NCLUSTERS
        else:
            raise NotImplemented
        self.__build_views__()

    def __build_views__(self):
        np.random.seed(self.seed)
        #random_state = np.random.randint(2 ** 16 - 1)
        random_state = self.seed
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)

        for viewname in self.data_views:
            X = self.data_views[viewname] # this is a sparse matrix!
            #km = KMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km = MiniBatchKMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km_labels = km.fit_predict(X)
            self.views["{0}_K{1}".format(viewname, self.NCLUSTERS[viewname])] = km_labels


"""
Caltech-7/20
"""
class CaltechN(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.logger = logger
        self.seed = seed
        mlabData = loadmat(dataset_dir)
        # self.name = self.__class__.__name__  # can be overrided if neccessary
        self.data_views = {"gabor":mlabData['X'][0,0],
                      "wm":mlabData['X'][0,1],
                      "centrist":mlabData['X'][0,2],
                      "hog":mlabData['X'][0,3],
                      "gist": mlabData['X'][0, 4],
                      "lbp":mlabData['X'][0,5]}
        self.views = {}
        self.labels = [lbl[0] for lbl in mlabData['Y']]
        self.name = 'Caltech-N'
        if (isinstance(NCLUSTERS, int)):
            # NCLUSTERS = {'tfidf':NCLUSTERS, 'lda':NCLUSTERS, 'skipgram':NCLUSTERS}
            self.NCLUSTERS = {}
            for v in self.data_views:
                # if not v in ["labels","dataset"]:
                self.NCLUSTERS[v] = NCLUSTERS
        elif isinstance(NCLUSTERS, dict):
            self.NCLUSTERS = NCLUSTERS
        else:
            raise NotImplemented
        self.__build_views__()

    def __build_views__(self):
        np.random.seed(self.seed)
        #random_state = np.random.randint(2 ** 16 - 1)
        random_state = self.seed
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)

        for viewname in self.data_views:
            X = self.data_views[viewname] # this is a sparse matrix!
            #km = KMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km = MiniBatchKMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km_labels = km.fit_predict(X)
            self.views["{0}_K{1}".format(viewname, self.NCLUSTERS[viewname])] = km_labels


"""
NUS-WIDE
"""
class NusWide(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.logger = logger
        self.seed = seed
        mlabData = loadmat(dataset_dir)
        # self.name = self.__class__.__name__  # can be overrided if neccessary
        self.data_views = {"ch":mlabData['X'][0,0],
                      "cm":mlabData['X'][0,1],
                      "corr":mlabData['X'][0,2],
                      "edh":mlabData['X'][0,3],
                      "wt": mlabData['X'][0, 4]}
        self.views = {}
        self.labels = [lbl[0] for lbl in mlabData['Y']]
        self.name = 'Nus-Wide'
        if (isinstance(NCLUSTERS, int)):
            # NCLUSTERS = {'tfidf':NCLUSTERS, 'lda':NCLUSTERS, 'skipgram':NCLUSTERS}
            self.NCLUSTERS = {}
            for v in self.data_views:
                # if not v in ["labels","dataset"]:
                self.NCLUSTERS[v] = NCLUSTERS
        elif isinstance(NCLUSTERS, dict):
            self.NCLUSTERS = NCLUSTERS
        else:
            raise NotImplemented
        self.__build_views__()

    def __build_views__(self):
        np.random.seed(self.seed)
        #random_state = np.random.randint(2 ** 16 - 1)
        random_state = self.seed
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)

        for viewname in self.data_views:
            X = self.data_views[viewname] # this is a sparse matrix!
            #km = KMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km = MiniBatchKMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km_labels = km.fit_predict(X)
            self.views["{0}_K{1}".format(viewname, self.NCLUSTERS[viewname])] = km_labels


"""
Handwritten
"""
class Handwritten(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.logger = logger
        self.seed = seed
        mlabData = loadmat(dataset_dir)
        # self.name = self.__class__.__name__  # can be overrided if neccessary
        self.data_views = {"pix":mlabData['X'][0,0],
                      "fou":mlabData['X'][0,1],
                      "fac":mlabData['X'][0,2],
                      "zer":mlabData['X'][0,3],
                      "kar": mlabData['X'][0, 4],
                      "mor":mlabData['X'][0,5]}
        self.views = {}
        self.labels = [lbl[0] for lbl in mlabData['Y']]
        self.name = 'Handwritten'
        if (isinstance(NCLUSTERS, int)):
            # NCLUSTERS = {'tfidf':NCLUSTERS, 'lda':NCLUSTERS, 'skipgram':NCLUSTERS}
            self.NCLUSTERS = {}
            for v in self.data_views:
                # if not v in ["labels","dataset"]:
                self.NCLUSTERS[v] = NCLUSTERS
        elif isinstance(NCLUSTERS, dict):
            self.NCLUSTERS = NCLUSTERS
        else:
            raise NotImplemented
        self.__build_views__()

    def __build_views__(self):
        np.random.seed(self.seed)
        #random_state = np.random.randint(2 ** 16 - 1)
        random_state = self.seed
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)

        for viewname in self.data_views:
            X = self.data_views[viewname] # this is a sparse matrix!
            #km = KMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km = MiniBatchKMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km_labels = km.fit_predict(X)
            self.views["{0}_K{1}".format(viewname, self.NCLUSTERS[viewname])] = km_labels

"""
Reuters5
"""
class Reuters5(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.logger = logger
        self.seed = seed
        mlabData = loadmat(dataset_dir)
        # self.name = self.__class__.__name__  # can be overrided if neccessary
        self.data_views = {"english":mlabData['X'][0,0],
                      "france":mlabData['X'][0,1],
                      "german":mlabData['X'][0,2],
                      "italian":mlabData['X'][0,3],
                      "spanish":mlabData['X'][0,4]}
        self.views = {}
        self.labels = [lbl[0] for lbl in mlabData['Y']]
        self.name = 'Reuters5'
        if (isinstance(NCLUSTERS, int)):
            # NCLUSTERS = {'tfidf':NCLUSTERS, 'lda':NCLUSTERS, 'skipgram':NCLUSTERS}
            self.NCLUSTERS = {}
            for v in self.data_views:
                # if not v in ["labels","dataset"]:
                self.NCLUSTERS[v] = NCLUSTERS
        elif isinstance(NCLUSTERS, dict):
            self.NCLUSTERS = NCLUSTERS
        else:
            raise NotImplemented
        self.__build_views__()

    def __build_views__(self):
        np.random.seed(self.seed)
        #random_state = np.random.randint(2 ** 16 - 1)
        random_state = self.seed
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)

        for viewname in self.data_views:
            X = self.data_views[viewname] # this is a sparse matrix!
            #km = KMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km = MiniBatchKMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km_labels = km.fit_predict(X)
            self.views["{0}_K{1}".format(viewname, self.NCLUSTERS[viewname])] = km_labels


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


if __name__ == '__main2__':
    rt5 = Reuters5("D:/mvdata/Reuters.mat", 6, 101)
    rt5.create_view(10)
    assess_datasource_views(rt5)

    hwt = Handwritten("D:/mvdata/handwritten.mat", 10, 101)
    hwt.create_view(5)
    hwt.create_view(15)
    assess_datasource_views(hwt)

    ct7 = CaltechN("D:/mvdata/Caltech101-7.mat", 7, 101)
    ct7.create_view(3)
    ct7.create_view(10)
    assess_datasource_views(ct7)

    ct20 = CaltechN("D:/mvdata/Caltech101-20.mat", 20, 101)
    ct20.create_view(10)
    ct20.create_view(15)
    assess_datasource_views(ct20)

    nusw = NusWide("D:/mvdata/NUSWIDEOBJ.mat", 31, 101)
    nusw.create_view(15)
    nusw.create_view(25)
    assess_datasource_views(nusw)

##########################
def obtain_doc_and_column_indexes(docsFname):
    doc_list = []
    doc_colums = {}
    with open(docsFname) as fp:
        cnt = 0
        while True:
            ln = fp.readline()
            if not ln:
                break
            doc_name = ln.strip()
            doc_list.append(doc_name)
            doc_colums[doc_name] = cnt
    return {'docList':doc_list, 'docCol':doc_colums}

if __name__ == '__main__':
    bbc_datadir = 'D:/mvdata/bbc-segment/bbc/'
    bbc_mat12_docs = obtain_doc_and_column_indexes('%sbbc_seg1of2.docs' % bbc_datadir)
    bbc_mat22_docs = obtain_doc_and_column_indexes('%sbbc_seg2of2.docs' % bbc_datadir)

    bbc_mat13_docs = obtain_doc_and_column_indexes('%sbbc_seg1of3.docs' % bbc_datadir)
    bbc_mat23_docs = obtain_doc_and_column_indexes('%sbbc_seg2of3.docs' % bbc_datadir)
    bbc_mat33_docs = obtain_doc_and_column_indexes('%sbbc_seg3of3.docs' % bbc_datadir)

    bbc_mat14_docs = obtain_doc_and_column_indexes('%sbbc_seg1of4.docs' % bbc_datadir)
    bbc_mat24_docs = obtain_doc_and_column_indexes('%sbbc_seg2of4.docs' % bbc_datadir)
    bbc_mat34_docs = obtain_doc_and_column_indexes('%sbbc_seg3of4.docs' % bbc_datadir)
    bbc_mat44_docs = obtain_doc_and_column_indexes('%sbbc_seg4of4.docs' % bbc_datadir)

    print('bbc-seg2:')
    print('view 1:',len(bbc_mat12_docs['docList']))
    print('view 2:', len(bbc_mat22_docs['docList']))
    bbc_seg2_commons = set(bbc_mat12_docs['docList']) & set(bbc_mat22_docs['docList'])
    print('common:', len(bbc_seg2_commons)) # docs in both views

    print('\nbbc-seg3:')
    print('view 1:', len(bbc_mat13_docs['docList']))
    print('view 2:', len(bbc_mat23_docs['docList']))
    print('view 3:', len(bbc_mat33_docs['docList']))
    bbc_seg3_commons = set(bbc_mat13_docs['docList']) & set(bbc_mat23_docs['docList']) & set(bbc_mat33_docs['docList'])
    print('common:', len(bbc_seg3_commons))  # docs in both views

    print('\nbbc-seg4:')
    print('view 1:',len(bbc_mat14_docs['docList']))
    print('view 2:',len(bbc_mat24_docs['docList']))
    print('view 3:',len(bbc_mat34_docs['docList']))
    print('view 4:',len(bbc_mat44_docs['docList']))
    bbc_seg4_commons = set(bbc_mat14_docs['docList']) & set(bbc_mat24_docs['docList']) & set(bbc_mat34_docs['docList']) & set(
        bbc_mat44_docs['docList'])
    print('common:', len(bbc_seg4_commons))  # docs in both views

    # Data matrices with  term frequencies
    bbc_mat12 = scipy.io.mmread('%sbbc_seg1of2.mtx' % bbc_datadir).tocsc()
    bbc_mat22 = scipy.io.mmread('%sbbc_seg2of2.mtx' % bbc_datadir).tocsc()
    v1_selected = np.array([bbc_mat12_docs['docCol'][doc] for doc in bbc_seg2_commons])
    v2_selected = np.array([bbc_mat22_docs['docCol'][doc] for doc in bbc_seg2_commons])
    bbc_mat12_curated = bbc_mat12[:,v1_selected]
    bbc_mat22_curated = bbc_mat22[:, v2_selected]
    print('\nBBC seg2:')
    print("Shape V1:",bbc_mat12_curated.shape[0], bbc_mat12_curated.shape[1])
    print("Shape V2:", bbc_mat22_curated.shape[0], bbc_mat22_curated.shape[1])


    bbc_mat13 = scipy.io.mmread('%sbbc_seg1of3.mtx' % bbc_datadir).tocsc()
    bbc_mat23 = scipy.io.mmread('%sbbc_seg2of3.mtx' % bbc_datadir).tocsc()
    bbc_mat33 = scipy.io.mmread('%sbbc_seg3of3.mtx' % bbc_datadir).tocsc()
    v1_selected = np.array([bbc_mat13_docs['docCol'][doc] for doc in bbc_seg3_commons])
    v2_selected = np.array([bbc_mat23_docs['docCol'][doc] for doc in bbc_seg3_commons])
    v3_selected = np.array([bbc_mat33_docs['docCol'][doc] for doc in bbc_seg3_commons])
    bbc_mat13_curated = bbc_mat13[:, v1_selected]
    bbc_mat23_curated = bbc_mat23[:, v2_selected]
    bbc_mat33_curated = bbc_mat33[:, v3_selected]
    print('\nBBC seg3:')
    print("Shape V1:",bbc_mat13_curated.shape[0], bbc_mat13_curated.shape[1])
    print("Shape V2:", bbc_mat23_curated.shape[0], bbc_mat23_curated.shape[1])
    print("Shape V3:", bbc_mat33_curated.shape[0], bbc_mat33_curated.shape[1])


    bbc_mat14 = scipy.io.mmread('%sbbc_seg1of4.mtx' % bbc_datadir).tocsc()
    bbc_mat24 = scipy.io.mmread('%sbbc_seg2of4.mtx' % bbc_datadir).tocsc()
    bbc_mat34 = scipy.io.mmread('%sbbc_seg3of4.mtx' % bbc_datadir).tocsc()
    bbc_mat44 = scipy.io.mmread('%sbbc_seg4of4.mtx' % bbc_datadir).tocsc()
    v1_selected = np.array([bbc_mat14_docs['docCol'][doc] for doc in bbc_seg4_commons])
    v2_selected = np.array([bbc_mat24_docs['docCol'][doc] for doc in bbc_seg4_commons])
    v3_selected = np.array([bbc_mat34_docs['docCol'][doc] for doc in bbc_seg4_commons])
    v4_selected = np.array([bbc_mat44_docs['docCol'][doc] for doc in bbc_seg4_commons])
    bbc_mat14_curated = bbc_mat14[:, v1_selected]
    bbc_mat24_curated = bbc_mat24[:, v2_selected]
    bbc_mat34_curated = bbc_mat34[:, v3_selected]
    bbc_mat44_curated = bbc_mat44[:, v4_selected]
    print('\nBBC seg4:')
    print("Shape V1:",bbc_mat14_curated.shape[0], bbc_mat14_curated.shape[1])
    print("Shape V2:", bbc_mat24_curated.shape[0], bbc_mat24_curated.shape[1])
    print("Shape V3:", bbc_mat34_curated.shape[0], bbc_mat34_curated.shape[1])
    print("Shape V4:", bbc_mat44_curated.shape[0], bbc_mat44_curated.shape[1])