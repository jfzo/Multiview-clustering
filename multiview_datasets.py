import logging
from logging_setup import logger

from data_source import DataViewGenerator

import numpy as np
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans, KMeans
import scipy
from scipy.io import loadmat

from tabulate import tabulate
import sklearn.metrics as evalfns


### Below... multiview datasets for the journal version.

"""
BBC
"""

class BBC(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.logger = logger
        self.seed = seed

        data_npz = np.load(dataset_dir, allow_pickle=True)
        # bbc3 = np.load('bbc_data_seg3.npz', allow_pickle=True)
        # bbc4 = np.load('bbc_data_seg4.npz', allow_pickle=True)
        # bbc2['X']
        # bbc2['Y']
        nviews = data_npz['X'].shape[1] # depends on the bbc segment
        self.data_views = {}
        for vi in range(nviews):
            self.data_views['v{0}'.format(vi)] = data_npz['X'][0, vi]

        self.views = {}
        self.labels = data_npz['Y']
        self.name = 'BBC-seg{0}'.format(nviews)
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

        for viewname in self.data_views:
            X = self.data_views[viewname] # this is a sparse matrix!
            #km = KMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km = MiniBatchKMeans(n_clusters=self.NCLUSTERS[viewname], init='k-means++', random_state=random_state)
            km_labels = km.fit_predict(X)
            self.views["{0}_K{1}".format(viewname, self.NCLUSTERS[viewname])] = km_labels

class BBC_seg2(BBC):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        dataset_dir = dataset_dir + "/bbc_data_seg2.npz"
        super(BBC_seg2, self).__init__(dataset_dir, NCLUSTERS, seed)

class BBC_seg3(BBC):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        dataset_dir = dataset_dir + "/bbc_data_seg3.npz"
        super(BBC_seg3, self).__init__(dataset_dir, NCLUSTERS, seed)

class BBC_seg4(BBC):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        dataset_dir = dataset_dir + "/bbc_data_seg4.npz"
        super(BBC_seg4, self).__init__(dataset_dir, NCLUSTERS, seed)

"""
Caltech-7/20
"""
class CaltechN(DataViewGenerator):
    def __init__(self, dataset_dir: str, NCLUSTERS: object, seed: int) -> None:
        self.logger = logger
        self.seed = seed
        mlabData = loadmat(dataset_dir+"/Caltech101-7.mat")
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
        mlabData = loadmat(dataset_dir+"/NUSWIDEOBJ.mat")
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
        mlabData = loadmat(dataset_dir+"/handwritten.mat")
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
        mlabData = loadmat(dataset_dir+"/Reuters.mat")
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
            cnt += 1
    return {'docList':doc_list, 'docCol':doc_colums}


def delete_empty_cols(csc_mat):
    """
    Very costful solution to the removal of features not present in the final document-term matrices.
    :param csc_mat: sparse matrix
    :return: Returns a CSR matrix identical to the one given as input but excluding the null columns.
    """
    cols_to_del = []
    for i in range(csc_mat.shape[1]):
        if csc_mat[:, i].nnz == 0:
            cols_to_del.append(i)

    cols_to_del.reverse()

    denseMat = csc_mat.toarray()
    for col in cols_to_del:
        denseMat = np.delete(denseMat, col, 1)

    return scipy.sparse.csr_matrix(denseMat)


def preprocess_bbc_data(bbc_datadir):
    """
    Preprocessing procedure employed to generate the datasets
    :param bbc_datadir: Location of segments
    :return:
    """
    # obtaining doc labels
    doc_label = dict()
    label_cnt = 0
    with open('%sbbc.clist' % bbc_datadir) as fp:
        while True:
            ln = fp.readline().strip()
            if not ln:
                break
            partsInLn = ln.split(":")
            label = partsInLn[0].strip()
            labeledDocs = partsInLn[1].strip().split(",")
            for doc in labeledDocs:
                #doc_label[doc] = label
                doc_label[doc.strip()] = label_cnt
            label_cnt += 1


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
    bbc_seg2_commons = list( set(bbc_mat12_docs['docList']) & set(bbc_mat22_docs['docList']) )
    print('common:', len(bbc_seg2_commons)) # docs in both views


    print('\nbbc-seg3:')
    print('view 1:', len(bbc_mat13_docs['docList']))
    print('view 2:', len(bbc_mat23_docs['docList']))
    print('view 3:', len(bbc_mat33_docs['docList']))
    bbc_seg3_commons = list( set(bbc_mat13_docs['docList']) & set(bbc_mat23_docs['docList']) & set(bbc_mat33_docs['docList']) )
    print('common:', len(bbc_seg3_commons))  # docs in both views

    print('\nbbc-seg4:')
    print('view 1:',len(bbc_mat14_docs['docList']))
    print('view 2:',len(bbc_mat24_docs['docList']))
    print('view 3:',len(bbc_mat34_docs['docList']))
    print('view 4:',len(bbc_mat44_docs['docList']))
    bbc_seg4_commons = list( set(bbc_mat14_docs['docList']) & set(bbc_mat24_docs['docList']) & set(bbc_mat34_docs['docList']) & set(
        bbc_mat44_docs['docList']) )
    print('common:', len(bbc_seg4_commons))  # docs in both views

    ## Data matrices with  term frequencies
    #
    bbc_mat12 = scipy.io.mmread('%sbbc_seg1of2.mtx' % bbc_datadir).tocsc()
    bbc_mat22 = scipy.io.mmread('%sbbc_seg2of2.mtx' % bbc_datadir).tocsc()
    v1_selected = np.array([bbc_mat12_docs['docCol'][doc] for doc in bbc_seg2_commons])
    v2_selected = np.array([bbc_mat22_docs['docCol'][doc] for doc in bbc_seg2_commons])
    seg2_labels = np.array([doc_label[doc] for doc in bbc_seg2_commons])
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
    seg3_labels = np.array([doc_label[doc] for doc in bbc_seg3_commons])
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
    seg4_labels = np.array([doc_label[doc] for doc in bbc_seg4_commons])
    bbc_mat14_curated = bbc_mat14[:, v1_selected]
    bbc_mat24_curated = bbc_mat24[:, v2_selected]
    bbc_mat34_curated = bbc_mat34[:, v3_selected]
    bbc_mat44_curated = bbc_mat44[:, v4_selected]
    print('\nBBC seg4:')
    print("Shape V1:",bbc_mat14_curated.shape[0], bbc_mat14_curated.shape[1])
    print("Shape V2:", bbc_mat24_curated.shape[0], bbc_mat24_curated.shape[1])
    print("Shape V3:", bbc_mat34_curated.shape[0], bbc_mat34_curated.shape[1])
    print("Shape V4:", bbc_mat44_curated.shape[0], bbc_mat44_curated.shape[1])

    # data structure to store all BBC data views
    bbc_data_views = {'seg-2':{'X':np.ndarray((1,2), dtype=np.ndarray),'Y':seg2_labels, 'docs':bbc_seg2_commons},
                'seg-3':{'X':np.ndarray((1,3), dtype=np.ndarray),'Y':seg3_labels, 'docs':bbc_seg3_commons},
                'seg-4':{'X':np.ndarray((1,4), dtype=np.ndarray),'Y':seg4_labels, 'docs':bbc_seg4_commons}
                }

    bbc_data_views['seg-2']['X'][0, 0] = delete_empty_cols(bbc_mat12_curated.transpose())
    bbc_data_views['seg-2']['X'][0, 1] = delete_empty_cols(bbc_mat22_curated.transpose())

    bbc_data_views['seg-3']['X'][0, 0] = delete_empty_cols(bbc_mat13_curated.transpose())
    bbc_data_views['seg-3']['X'][0, 1] = delete_empty_cols(bbc_mat23_curated.transpose())
    bbc_data_views['seg-3']['X'][0, 2] = delete_empty_cols(bbc_mat33_curated.transpose())

    bbc_data_views['seg-4']['X'][0, 0] = delete_empty_cols(bbc_mat14_curated.transpose())
    bbc_data_views['seg-4']['X'][0, 1] = delete_empty_cols(bbc_mat24_curated.transpose())
    bbc_data_views['seg-4']['X'][0, 2] = delete_empty_cols(bbc_mat34_curated.transpose())
    bbc_data_views['seg-4']['X'][0, 3] = delete_empty_cols(bbc_mat44_curated.transpose())

    np.savez('bbc_data_seg2.npz', **bbc_data_views['seg-2'], allow_pickle=True)
    np.savez('bbc_data_seg3.npz', **bbc_data_views['seg-3'], allow_pickle=True)
    np.savez('bbc_data_seg4.npz', **bbc_data_views['seg-4'], allow_pickle=True)


if __name__ == '__main__':
    #preprocess_bbc_data('D:/multi-view-data/bbc-segment/bbc/')
    #preprocess_bbc_data('D:/Google Drive/Research - Multiview and Collaborative Clustering/data/bbc/')
    print("Main method execution...")
