from multiview_datasets import Handwritten, NusWide, CaltechN, BBC, Reuters5
from data_source import assess_datasource_views

if __name__ == '__main__':
    rt5 = Reuters5("D:/multi-view-data/Reuters.mat", 6, 101)
    #rt5.create_view(10)
    print("\nReuters (5 views)\n")
    assess_datasource_views(rt5)

    hwt = Handwritten("D:/multi-view-data/handwritten.mat", 10, 101)
    #hwt.create_view(5)
    #hwt.create_view(15)
    print("\nHandwritten (6 views)\n")
    assess_datasource_views(hwt)

    ct7 = CaltechN("D:/multi-view-data/Caltech101-7.mat", 7, 101)
    #ct7.create_view(3)
    #ct7.create_view(10)
    print("\nCaltech-7 (6 views)\n")
    assess_datasource_views(ct7)

    ct20 = CaltechN("D:/multi-view-data/Caltech101-20.mat", 20, 101)
    #ct20.create_view(10)
    #ct20.create_view(15)
    print("\nCaltech-20 (6 views)\n")
    assess_datasource_views(ct20)

    nusw = NusWide("D:/multi-view-data/NUSWIDEOBJ.mat", 31, 101)
    #nusw.create_view(15)
    #nusw.create_view(25)
    print("\nNus-Wide (5 views)\n")
    assess_datasource_views(nusw)

    bbc_s2 = BBC('D:/multi-view-data/bbc_data_seg2.npz', 5, 101)
    #bbc_s2.create_view(3)
    #bbc_s2.create_view(10)
    print("\nBBC-2 (2 views)\n")
    assess_datasource_views(bbc_s2)

    bbc_s3 = BBC('D:/multi-view-data/bbc_data_seg3.npz', 5, 101)
    #bbc_s3.create_view(3)
    #bbc_s3.create_view(10)
    print("\nBBC-3 (3 views)\n")
    assess_datasource_views(bbc_s3)

    bbc_s4 = BBC('D:/multi-view-data/bbc_data_seg4.npz', 5, 101)
    #bbc_s4.create_view(3)
    #bbc_s4.create_view(10)
    print("\nBBC-4 (4 views)\n")
    assess_datasource_views(bbc_s4)