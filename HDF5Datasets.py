from logging_setup import logger
import os
import h5py
import numpy as np
from data_source import DataViewGenerator
from utils import sparseMatFromCluto
from scipy.sparse import csr_matrix

"""
Some usage examples:
from sklearn import metrics
metrics.adjusted_rand_score(rtds.get_real_labels(), rtds.get_views()['rt_german'])
metrics.adjusted_mutual_info_score(rtds.get_real_labels(), rtds.get_views()['rt_german'])
metrics.homogeneity_completeness_v_measure(rtds.get_real_labels(), rtds.get_views()['rt_german'])
---
# The Silhouette Coefficient is defined for each sample. It's composed of two scores:
# a: The mean distance between a sample and all other points in the same class.
# b: The mean distance between a sample and all other points in the next nearest cluster.
# A higher Silhouette Coefficient score relates to a model with better defined clusters
metrics.silhouette_score(rtds.get_data_views()['rt_german'], rtds.get_views()['rt_german'], metric='cosine')
metrics.silhouette_score(rtds.get_data_views()['rt_german'], rtds.get_real_labels(), metric='cosine')
---
# The index is the ratio of the sum of between-clusters dispersion and of inter-cluster dispersion for all clusters
# ...(where dispersion is defined as the sum of distances squared)
# A higher Calinski-Harabasz score relates to a model with better defined clusters
metrics.calinski_harabasz_score(rtds.get_data_views()['rt_german'].toarray(), rtds.get_views()['rt_german'])
metrics.calinski_harabasz_score(rtds.get_data_views()['rt_german'].toarray(), rtds.get_real_labels())
---
# This index signifies the average ‘similarity’ between clusters 
# A lower Davies-Bouldin index relates to a model with better separation between the clusters.
metrics.davies_bouldin_score(rtds.get_data_views()['rt_german'].toarray(), rtds.get_views()['rt_german'])
metrics.davies_bouldin_score(rtds.get_data_views()['rt_german'].toarray(), rtds.get_real_labels())
"""
class HDF5Adapter (DataViewGenerator):
    def __init__(self, name, path):
        self.logger = logger
        self.path = path
        self.name = name
        self.views = {}
        self.data_views = None
        self.labels = None
        self.__build_views__()

    def __build_views__(self):
        f = h5py.File(self.path, 'r')
        self.labels = f['labels'][()]
        #csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
        viewKeys = f['views'].keys()
        self.data_views = {}
        for vw in viewKeys:
            self.views[vw] = f['views'][vw]['labels'][()]
            data = f['views'][vw]['csr-format']['data'][()]
            indices = f['views'][vw]['csr-format']['indices'][()]
            indptr = f['views'][vw]['csr-format']['indptr'][()]
            self.shape = f['views'][vw]['csr-format']['shape'][()]
            self.data_views[vw] = csr_matrix((data, indices, indptr), shape=self.shape)




class BBCseg4(HDF5Adapter):
    def __init__(self, path='.', fname='bbc-seg4.hdf5'):
        super().__init__("BBC-seg4", path + os.sep + fname)

class Caltech20(HDF5Adapter):
    def __init__(self, path='.', fname='caltech-20.hdf5'):
        super().__init__("Caltech-20", path + os.sep + fname)

class Handwritten(HDF5Adapter):
    def __init__(self, path='.', fname='handwritten.hdf5'):
        super().__init__("Handwritten", path + os.sep + fname)

class NusWide(HDF5Adapter):
    def __init__(self, path='.', fname='nusWide.hdf5'):
        super().__init__("Nus-Wide", path + os.sep + fname)

class Reuters5(HDF5Adapter):
    def __init__(self, path='.', fname='reuters.hdf5'):
        super().__init__("Reuters-5", path + os.sep + fname)


def createHDF5Datasets(inDir):
    #inDir = '/home/juan/Documentos/sparse-multiview-data/'
    dirContent = os.listdir(inDir)
    for dirItem in dirContent:
        fullPathFsItem = inDir + "/" + dirItem # specific dataset folder
        if os.path.isdir(fullPathFsItem) and not dirItem.startswith("."):
            print(dirItem)
            # [within the folder] several files : {AS_FOLDER}.labels(1) {prefix}[.dat | .clustering.XX](* for each view)
            dsDirContent = os.listdir(fullPathFsItem)
            print('Creating file {0}/{1}.hdf5'.format(inDir,dirItem))
            hdfDS = h5py.File('{0}/{1}.hdf5'.format(inDir,dirItem), 'w')
            hdfDS.attrs['date_'] = '5.11.2020'
            hdfDS.attrs['creator_'] = 'Juan Zamora O.'
            #hdfDS.create_group('labels')
            hdfDS.create_group('views')
            # each .dat and clustering files
            inputDataFiles = [(fn,cfn) for fn in dsDirContent if fn.endswith(".dat")
                         for cfn in dsDirContent if '.clustering.' in cfn and fn in cfn]
            print('Iterating over',inputDataFiles)
            for datFn, clustResFn in inputDataFiles:
                k = datFn.replace(".dat","")
                vwLabels = np.loadtxt('{0}/{1}'.format(fullPathFsItem,clustResFn), dtype=np.int8) #clustering result associated to the .dat file
                vwData = sparseMatFromCluto('{0}/{1}'.format(fullPathFsItem, datFn), sparseFmt = True)
                hdfDS['views'].create_group(k)
                #hdfDS['views'][k].create_group('labels')
                #hdfDS['views'][k].create_group('data')
                hdfDS['views'][k]['labels'] = vwLabels
                #hdfDS['views'][k]['data'] = vwData.todense()
                hdfDS['views'][k].create_group('csr-format')
                hdfDS['views'][k]['csr-format']['data'] = vwData.data
                hdfDS['views'][k]['csr-format']['indices'] = vwData.indices
                hdfDS['views'][k]['csr-format']['indptr'] = vwData.indptr
                hdfDS['views'][k]['csr-format']['shape'] = vwData.shape

            hdfDS['labels'] = np.loadtxt('{0}/{1}.labels'.format(fullPathFsItem,dirItem), dtype=np.int8)
            hdfDS.close()
