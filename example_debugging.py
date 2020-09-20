from original_ncf import NCFwR
from multiview_datasets import Handwritten, NusWide

def test_NCFwR():
    #hndwrDS = Handwritten('D:/mvdata', 6, 65578)
    #met_name = met_op.getName()
    #met_op.setInputPartitions(hndwrDS.get_views())
    nswDS = NusWide('/mnt/windows/mvdata', 6, 65578)
    met_op = NCFwR(args={'seed': 65578, 'number_random_partitions': 50})
    met_op.setInputPartitions(nswDS.get_views())
    cnsPart1 = met_op.run()

if __name__ == '__main__':
    test_NCFwR()
