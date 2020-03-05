from sklearn import preprocessing
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import cluster

def __gen_hyperedge(labels):
    lb = preprocessing.LabelBinarizer()
    return lb.fit_transform(labels)

def __gen_hypergraph(Views):
    hedges = [__gen_hyperedge(v) for v in Views]
    return np.hstack(hedges)

def CBSP_similarity(Views):
    H = __gen_hypergraph(Views)
    return (1/len(Views)) * H.dot(H.transpose())


def build_consensus_distance(V):
    n = len(V[0])
    D = np.zeros((n,n))
    for v in V:
        H = __gen_hyperedge(v)
        D += (1 - H.dot(H.transpose()))
    return D

def ensemble_similarity(V):
    # fixing views labels
    le = preprocessing.LabelEncoder()
    Views = [le.fit_transform(v) for v in V]
    
    S_H = CBSP_similarity(Views)
    D_PDM = build_consensus_distance(Views)
    S_PDM = cosine_similarity(D_PDM)
    
    return 0.5 * (S_H + S_PDM)


def ensemble_clustering(Views, K):
    S = ensemble_similarity(Views)
    D = 1 - S
    
    single = cluster.AgglomerativeClustering(n_clusters=K, linkage='single', affinity='precomputed')
    single.fit(D)
    return single.labels_.astype(np.int)


####
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
  