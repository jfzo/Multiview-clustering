import logging
from logging_setup import logger

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
from sklearn import preprocessing
import csv
import scipy.sparse
import random

def K_int(n):
    return np.floor(np.log2(n))

def compute_solutions_complexity(S1, S2, K1, K2, labels=None):
    """
    Computes kolmogorov complexity between two clusterings.
    S1, S2 : numpy 1d length-n arrays contaning labels of each data point.
    Cluster ids in S1 range from 0 to (K1-1)
    Cluster ids in S2 range from 0 to (K2-1)
    K1, K2 : number of clusters in each partitioning.
    """
    # K(S1 | S2)
    # TODO: Normalize labels both solutions to be in ranges [0,K1[ and [0,K2[ respectively.
    assert check_correlative_cluster_labels(S1)
    assert check_correlative_cluster_labels(S2)
    
    #cnf_matrix = confusion_matrix(S2, S1) # y_true, y_pred
    # rows in confusion_mat are associated to S2
    # cols in confusion_mat are associated to S1
    # argmax(1) returns the column index where the max agreement is for each row.
    
    # I changed because it seems to me that it was incorrectly inverted.
    cnf_matrix = confusion_matrix(S1, S2, labels=labels) # y_true, y_pred
    
    
    rules = cnf_matrix.argmax(1) # indexes of the maximum in each row (maximum match between two clusterings) --> in range [0,K1[
    complexity = K2 * (K_int(K1) + K_int(K2))
    
    # Search of exceptions
    exceptions = {}
    for i in range(S1.shape[0]): # S1.shape[0] denotes number of elements in the dataset
        # In order to use the following expression, both clusterings would have to be normalized with labels in [0,K1[ and [0,K2[
        if S1[i] != rules[S2[i]]: # if cluster assigned by localsm1 does not match the mayority group in localsm2, add an exception!
            exceptions[i] = S1[i] # data point as key and conflicting cluster id as value
            complexity += K_int(S1.shape[0]) + K_int(K1) # for each exception: log2(N)+log2(K_1)
    return rules, exceptions, complexity


def fix_cluster_labels(cl):
    u_c = np.unique(cl)
    c_mapping = dict(zip(u_c, list(range(len(u_c))))) # makes consecutive labels
    cpcl = cl.copy()
    
    for i in c_mapping:
        cpcl[cl == i] = c_mapping[i]
    return cpcl


def random_partition(Kmax, n):
    """
    Generates a assignment of `n` objects into a random number of clusters.
    Each objects is assigned to a cluster id in [0, Kmax] (extremes included).
    """
    rnd_c = [np.random.randint(0, Kmax + 1) for i in range(n)]
    
    ord_c = np.unique(rnd_c)
    rnd_c = np.array([np.where(ord_c == i)[0][0] for i in rnd_c])

    return rnd_c


def check_correlative_cluster_labels(S): 
    """
    Identifies if the clusters start from 0 and are correlative.
    This is very importante in order to employ the confusion matrix between two solutions.
    """
    # A solution vector is given
    unique_labels = np.unique(S[np.where(S>=0)[0]])
    return np.sum(np.sort(unique_labels) == np.array(range(len(unique_labels)))) == len(unique_labels)


def agreement_measure(P1, P2):
    """
    Jaccard Similarity J (A,B) = | Intersection (A,B) | /
                                    | Union (A,B) |
    """
    intersection_cardinality = len(set(P1).intersection(set(P2)))
    union_cardinality = len(set(P1).union(set(P2)))
    return intersection_cardinality / float(union_cardinality)    


def N_p_V(p, B):
    """
    N_p(B)
    p: integer denoting a specific point
    B: array with a label for each point (its id) denoting a View0
    Given a point p and a view B
    returns the partition (a set) that contains p (it is assumed that p has a valid label).
    """
    assert(B[p] > -1)
    # all points having the same cluster label as p
    return set(np.where(B == B[p])[0]) 
    #for partition in B:
    #    if p in partition:
    #        return partition
    #return None

def max_agreement_partition(P, B):
    """
    Phi_B(P):
    P: A set of integers denoting the points contained in the partition
    B: array with a label for each point (its id) denoting a View
    returns the partition (its id) from view B that has
    the highest agreement.
    
    Note: This function considers only the points having a label starting at 0.
    """
    # obtain all the different labels in B(!= -1)
    available_labels = np.unique(B[np.where(B > -1)[0]])
    # for each label, obtain a set of points having that label in B and measure agreement
    return np.argmax([agreement_measure(P, set(np.where(B == l)[0]) ) for l in available_labels] )
        

def omega(pi_, p, A, epweights_A):
    """
    pi_ : A set of integers denoting the points contained in the partition
    p: integer denoting a specific point
    A: array with a label for each point (its id) denoting a View 
    epweights_A : Dictionary with point weights for each cluster in A
    Weight function that measures the membership of point p in
    partition pi_C of view C given the status of p in view A.
    """
    # 1st check if this is the 1st time the point is marked 
    if A[p] != -1: 
        return agreement_measure(pi_, N_p_V(p, A))
    # otherwise...
    Phi_A_pi = max_agreement_partition(pi_, A)
    
    #logger.debug(Phi_A_pi)
    #logger.debug(list(epweights_A.keys()))

    # epweights_A must contain the tuple due to the former mark of p in A
    return agreement_measure(pi_, set(np.where(A == Phi_A_pi)[0])) * epweights_A[(p, Phi_A_pi)]
    

# evaluation


def cluster_purity(r, labels_pred, labels_true):
    """
    Computes the class purity for a specific group label.
    :param r: Specific group label.
    :param labels_pred: Predicted labels
    :param labels_true: Ground truth labels
    :return: class purity
    """
    r_items = np.where(labels_pred == r)[0]
    n_r = len(r_items)
    
    unique_items_lbls, lbls_cnts = np.unique(labels_true[r_items], return_counts=True)
    return (1/n_r)*np.max(lbls_cnts)

def Purity(labels_pred, labels_true):
    """
    Computes the average purity between all the clusters.
    :param labels_pred: Predicted labels
    :param labels_true: Ground truth labels
    :return: Average purity (higher the better!)
    """
    L_pred, cnt_pred= np.unique(labels_pred, return_counts=True)
    n = len(labels_pred)
    sum_E = 0
    for r in range(len(L_pred)):
        cp_r = cluster_purity(L_pred[r], labels_pred, labels_true)
        n_r = cnt_pred[r]
        sum_E += (n_r/n)*cp_r
    return sum_E
    
def cluster_entropy(r, labels_pred, labels_true):
    """
    Computes the class entropy for a specific group label.
    :param r: Specific group label.
    :param labels_pred: Predicted labels
    :param labels_true: Ground truth labels
    :return: class entropy
    """
    L_true= np.unique(labels_true)
    q = len(L_true)
    # get items with label r
    r_items = np.where(labels_pred == r)[0]
    n_r = len(r_items)
    
    items_l, cnts = np.unique(labels_true[r_items], return_counts=True)
    # for each item label i compute
    sum_ent = 0
    for i in range(len(items_l)): 
        ni_r = cnts[i]
        sum_ent += (ni_r/n_r)*np.log(ni_r/n_r) 
    
    return -1/(np.log(q))*(sum_ent)
    

def Entropy(labels_pred, labels_true):
    """
    Computes the average entropy between all the clusters.
    :param labels_pred: Predicted labels
    :param labels_true: Ground truth labels
    :return: Average entropy (lower the better!)
    """
    L_pred, cnt_pred= np.unique(labels_pred, return_counts=True)
    n = len(labels_pred)
    sum_E = 0
    for r in range(len(L_pred)):
        ce_r = cluster_entropy(L_pred[r], labels_pred, labels_true)
        n_r = cnt_pred[r]
        sum_E += (n_r/n)*ce_r
    return sum_E

if __name__ == '__main__':
    print(random_partition(5, 10))