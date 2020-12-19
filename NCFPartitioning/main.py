#import networkx as nx
#import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy import linalg
# This is a sample Python script.
import glob
import sys
from datetime import datetime

def detectPostfixesInDir(fullPath, dirName):
    fnLst = glob.glob('{0}/{1}/*labels_*.out'.format(fullPath, dirName))
    # labels_NCF_DS_
    postFixes = [f.split("/")[-1].replace('labels_NCF_DS_','').replace('.out','') for f in fnLst]
    return postFixes

"""
def graphTest():
    G = nx.Graph()
    G.add_node(1, name="n1")
    G.add_node(4, name="n4")
    G.add_node(6, name="n6")

    G.add_edge(1, 6, weight=0.4)
    G.add_edges_from([(1, 4), (4, 6)], weight=0.9)

    # Create positions of all nodes and save them
    pos = nx.spring_layout(G)
    # Draw the graph according to node positions
    nx.draw(G, pos, with_labels=True)
    # Create edge labels
    labels = {e: str(G.edges[e]['weight']) for e in G.edges}
    # Draw edge labels according to node positions
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    #plt.show()
# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
"""

def _loadSparseCSRFromDisk(fullPath):
    spD = None
    with open(fullPath, 'rb') as f:
        data = np.load(f)
        indices = np.load(f)
        indptr = np.load(f)
        spD = csr_matrix((data, indices, indptr))
        # convert this nxk matrix into a similarity.
    return spD

def generateSimilarityFromCSR(fullPath):
    spD = _loadSparseCSRFromDisk(fullPath)
    dotP = spD.dot(spD.transpose())

    return dotP


def loadSimilarityAsSparseCOO(fullPath):
    D = np.loadtxt(fullPath, delimiter=',')
    S = D.dot(D.transpose())
    spS = coo_matrix(S)
    return spS

"""
def similarityAsGraph(S):
    n = S.shape[0]
    G = nx.Graph()
    _ = [G.add_node(i, name="n{0}".format(i)) for i in range(n)]
    # processing edges
    for i in range(S.nnz):
        w = S.data[i]
        G.add_edge(S.row[i], S.col[i], weight=w)
    return G
"""

def getLabelsFromFile(fullPath):
    v = np.loadtxt(fullPath, delimiter=',', dtype=np.int)
    return v

# Press the green button in the gutter to run the script.
def main_eval_similarities_with_cluto(datasets, NClusters, use_custom_similarity, ctoptions):
    """
    Improves and Evaluates NCF over a dataset using the final cluster membership matrices obtained
    after several runs.
    The results are averaged and printed to a file.
    :param datasets:
    :param NClusters:
    :param use_custom_similarity: If True, a similarity matrix is computed and passed to Cluto (scluster input). Otherwise, membership matrix is passed straight to Cluto (vcluster input)
    :param ctoptions: Cluto options.
    :return: None
    """
    from pycluto import cluto_scluster, cluto_vcluster
    from clustering_evaluation import computeEvaluationMeasures, MEASURES
    from tabulate import tabulate


    #path = '.'
    path = '/home/juan/Insync/juan.zamora@pucv.cl/Google Drive/Research - Multiview and Collaborative Clustering/code/handwritten_tezt/similarity_matrices_after_merging'

    #datasets = ['BBC-seg4']#['BBC-seg4', 'Caltech', 'handwritten', 'NusWide']
    #NClusters = [5]#[5,20,10,31]

    for i in range(len(datasets)):
        dsDirPath = datasets[i]
        K = NClusters[i]

        postFixesDS = detectPostfixesInDir(path, dsDirPath)

        #ctoptions = "-clmethod=graph -crfun=i1 -cstype=large -nnbrs=40 -grmodel=sd"

        #ctoptions = "-seed=101 -clmethod=graph -crfun=g1p -colmodel=none -rowmodel=none -sim=cos -cstype=best -nnbrs=50 -grmodel=ad"
        ## BEST OPTION FOR BBC-seg4:  F1:0.358
        #ctoptions = "-clmethod=graph -crfun=g1p -colmodel=none -rowmodel=none -sim=cos -cstype=best -nnbrs=50 -grmodel=ad"
        # ...
        MEASURES = ['E', 'P', 'F1', 'ACC', 'NMI', 'PREC', 'REC', 'ARI']
        scores_NCFwR = dict(zip(MEASURES, [[] for _ in range(len(MEASURES))]))
        scores_NCFSIM = dict(zip(MEASURES, [[] for _ in range(len(MEASURES))]))
        #runPostfix = 'Nov262020.182009.out' # pull from the directory !
        for runPostfix in postFixesDS:
            trueLblsFname = "{0}/{1}/{2}".format(path, dsDirPath, 'labels.true')
            ncfPredLblsFname = "{0}/{1}/{2}{3}.out".format(path, dsDirPath, 'labels_NCF_DS_', runPostfix)
            #simFname = "{0}/{1}/{2}{3}.out".format(path, dsDirPath, 'simmatrix_DS_', runPostfix)
            simFname = "{0}/{1}/{2}{3}_csr.npy".format(path, dsDirPath, 'simmatrix_DS_', runPostfix)

            print("Processing {0}".format(simFname))
            trueL = getLabelsFromFile(trueLblsFname)
            ncfL = getLabelsFromFile(ncfPredLblsFname)

            #S = generateSimilarityFromCSR(simFname)
            # print("clustering the similarity graph...")
            #S = loadSimilarityAsSparseCOO(simFname)
            ##S = S.todense()

            predictedL = None

            if use_custom_similarity:
                S = generateSimilarityFromCSR(simFname)
                ##
                ##simFname = "{0}/{1}/{2}{3}.out".format(path, dsDirPath, 'simmatrix_DS_', runPostfix)
                ##S = loadSimilarityAsSparseCOO(simFname).todense()

                predictedL = cluto_scluster(S, K, delete_temporary=True,
                                            CLUTOV_CMD='/home/juan/cluto-2.1.2/Linux-i686-openmp/scluster',
                                            clutoOptions=ctoptions)
            else:
                D = _loadSparseCSRFromDisk(simFname)
                predictedL = cluto_vcluster(D, K, delete_temporary=True,
                                            CLUTOV_CMD='/home/juan/cluto-2.1.2/Linux-i686-openmp/vcluster',
                                            clutoOptions = ctoptions)

            #  /home/juan/cluto-2.1.2/Linux-x86_64/
            #print(scores_table(trueL, ncfL))
            sNCFwR = computeEvaluationMeasures(trueL, ncfL)
            #getTabularMeasures(trueL, ncfL)
            #print(scores_table(trueL, predictedL))
            sNCFSIM = computeEvaluationMeasures(trueL, predictedL)

            for msre in sNCFwR:
                scores_NCFwR[msre].append(sNCFwR[msre])
                scores_NCFSIM[msre].append(sNCFSIM[msre])
        
         
        now = datetime.now()
        outputfile = "clustering_%s_%s.txt" % (dsDirPath,now.strftime("%b%d%Y.%H%M%S"))
        out = open(outputfile, 'w')
        out.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% {0} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n".format(dsDirPath))
        out.write("########################### NCFwR ###########################\n")

        table = [[np.mean(scores_NCFwR[m]) for m in MEASURES]]
        out.write(tabulate(table, headers=MEASURES, tablefmt="fancy_grid"))
        out.write('\n')
        out.write("########################### NCFwR ###########################\n")
        out.write(tabulate(table, headers=MEASURES, tablefmt="latex"))
        out.write('\n')

        print("########################### NCFwR ###########################")
        print(tabulate(table, headers=MEASURES, tablefmt="latex"))

        out.write("########################### NCFSim ###########################\n")
        table = [[np.mean(scores_NCFSIM[m]) for m in MEASURES]]
        out.write(tabulate(table, headers=MEASURES, tablefmt="fancy_grid"))
        out.write('\n')
        out.write(tabulate(table, headers=MEASURES, tablefmt="latex"))

        print("########################### NCFSim ###########################")
        print(tabulate(table, headers=MEASURES, tablefmt="latex"))

        print("\nFile {0} written.".format(outputfile))

    """
    for i in range(S.shape[0]):
        S[i, i] = S[i, i] + 1e-5

    L = linalg.cholesky(S, lower=True)
    Shat = S.dot(linalg.inv(L).transpose())
    predictedL2 = cluto_scluster(Shat, 5, CLUTOV_CMD='/home/juan/cluto-2.1.2/Linux-x86_64/scluster')
    print("########################### NCFSim2 ###########################")
    print(scores_table(trueL, predictedL2))
    """



    #G = similarityAsGraph(S)

    print("end!")

def main_build_csr_from_dense_in_disk():
    from pycluto import convert_dense_csv_to_csr_fmt
    from clustering_evaluation import computeEvaluationMeasures, MEASURES
    from tabulate import tabulate

    # path = '.'
    path = '/home/juan/Insync/juan.zamora@pucv.cl/Google Drive/Research - Multiview and Collaborative Clustering/code/handwritten_tezt/similarity_matrices_after_merging'

    datasets = ['NusWide']  # ['BBC-seg4', 'Caltech', 'handwritten', 'NusWide']

    for i in range(len(datasets)):
        dsDirPath = datasets[i]
        postFixesDS = detectPostfixesInDir(path, dsDirPath)

        # runPostfix = 'Nov262020.182009.out' # pull from the directory !
        for runPostfix in postFixesDS:
            simFname = "{0}/{1}/{2}{3}.out".format(path, dsDirPath, 'simmatrix_DS_', runPostfix)

            print("Processing {0}".format(simFname))
            convert_dense_csv_to_csr_fmt(simFname)
            #S = loadSimilarityAsSparseCOO(simFname)

    """
    for i in range(S.shape[0]):
        S[i, i] = S[i, i] + 1e-5

    L = linalg.cholesky(S, lower=True)
    Shat = S.dot(linalg.inv(L).transpose())
    predictedL2 = cluto_scluster(Shat, 5, CLUTOV_CMD='/home/juan/cluto-2.1.2/Linux-x86_64/scluster')
    print("########################### NCFSim2 ###########################")
    print(scores_table(trueL, predictedL2))
    """

    # G = similarityAsGraph(S)

    print("end!")

if __name__ == '__main__':
    #main_build_csr_from_dense_in_disk()
    #'BBC-seg4']  # ['BBC-seg4', 'Caltech', 'handwritten', 'NusWide']
    # NClusters = [5]#[5,20,10,31]

    # BBC-seg4 (1)
    main_eval_similarities_with_cluto(['BBC-seg4'], [5],
                                      use_custom_similarity=True,
                                      #ctoptions = "-crfun=g1 -clmethod=graph -cstype=large -nnbrs=20 -grmodel=sd")
                                      ctoptions = "-clmethod=graph -crfun=g1p -cstype=large -nnbrs=40 -grmodel=sd")

    # BBC-seg4 (2)
    #main_eval_similarities_with_cluto(['BBC-seg4'], [5],
    #                                  use_custom_similarity=False,
    #                                  ctoptions="-seed=23292 -clmethod=graph -crfun=g1 -colmodel=none -rowmodel=none -sim=cos -cstype=best -nnbrs=50 -grmodel=ad")
