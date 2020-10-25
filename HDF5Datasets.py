from logging_setup import logger
import os
import h5py
import numpy as np
from data_source import DataViewGenerator



class HDF5Adapter (DataViewGenerator):
    def __init__(self, name, path):
        self.logger = logger
        self.path = path
        self.name = name
        self.views = {}
        self.labels = None
        self.__build_views__()

    def __build_views__(self):
        f = h5py.File(self.path, 'r')
        self.labels = f['true']['true'][()]
        viewKeys = f['view'].keys()
        for vw in viewKeys:
            self.views[vw] = f['view'][vw][()]

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
