from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, precision_score, recall_score, adjusted_rand_score
import numpy as np
from tabulate import tabulate

MEASURES = ['E', 'P', 'F1', 'ACC', 'NMI', 'PREC', 'REC', 'ARI']


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
    return (1 / n_r) * np.max(lbls_cnts)


def Purity(labels_pred, labels_true):
    """
    Computes the average purity between all the clusters.
    :param labels_pred: Predicted labels
    :param labels_true: Ground truth labels
    :return: Average purity (higher the better!)
    """
    L_pred, cnt_pred = np.unique(labels_pred, return_counts=True)
    n = len(labels_pred)
    sum_E = 0
    for r in range(len(L_pred)):
        cp_r = cluster_purity(L_pred[r], labels_pred, labels_true)
        n_r = cnt_pred[r]
        sum_E += (n_r / n) * cp_r
    return sum_E

def cluster_entropy(r, labels_pred, labels_true):
    """
    Computes the class entropy for a specific group label.
    :param r: Specific group label.
    :param labels_pred: Predicted labels
    :param labels_true: Ground truth labels
    :return: class entropy
    """
    L_true = np.unique(labels_true)
    q = len(L_true)
    # get items with label rloadmat
    r_items = np.where(labels_pred == r)[0]
    n_r = len(r_items)

    items_l, cnts = np.unique(labels_true[r_items], return_counts=True)
    # for each item label i compute
    sum_ent = 0
    for i in range(len(items_l)):
        ni_r = cnts[i]
        sum_ent += (ni_r / n_r) * np.log(ni_r / n_r)

    return -1 / (np.log(q)) * (sum_ent)


def Entropy(labels_pred, labels_true):
    """
    Computes the average entropy between all the clusters.
    :param labels_pred: Predicted labels
    :param labels_true: Ground truth labels
    :return: Average entropy (lower the better!)
    """
    L_pred, cnt_pred = np.unique(labels_pred, return_counts=True)
    n = len(labels_pred)
    sum_E = 0
    for r in range(len(L_pred)):
        ce_r = cluster_entropy(L_pred[r], labels_pred, labels_true)
        n_r = cnt_pred[r]
        sum_E += (n_r / n) * ce_r
    return sum_E


# f1_score
def F1Score(labels_pred, labels_true):
    return f1_score(labels_true, labels_pred, average='weighted')


# accuracy_score
def ACCScore(labels_pred, labels_true):
    return accuracy_score(labels_true, labels_pred)


# normalized_mutual_info_score
def NMIScore(labels_pred, labels_true):
    return normalized_mutual_info_score(labels_true, labels_pred)


# precision_score
def PRECScore(labels_pred, labels_true):
    return precision_score(labels_true, labels_pred, average='weighted')


# recall_score
def RECScore(labels_pred, labels_true):
    return recall_score(labels_true, labels_pred, average='weighted')


# adjusted_rand_score
def ARIScore(labels_pred, labels_true):
    return adjusted_rand_score(labels_true, labels_pred)

def computeEvaluationMeasures(trueL, predictedL):
    v_E = Entropy(predictedL, trueL)
    v_P = Purity(predictedL, trueL)
    v_F1 = F1Score(predictedL, trueL)
    v_ACC = ACCScore(predictedL, trueL)
    v_NMI = NMIScore(predictedL, trueL)
    v_PREC = PRECScore(predictedL, trueL)
    v_REC = RECScore(predictedL, trueL)
    v_ARI = ARIScore(predictedL, trueL)
    #header = ['E', 'P', 'F1', 'ACC', 'NMI', 'PREC', 'REC', 'ARI']
    return dict(zip(MEASURES, (v_E, v_P, v_F1, v_ACC, v_NMI, v_PREC, v_REC, v_ARI)))

def getTabularMeasures(trueL, predictedL, tabFmt="fancy_grid"):
    #header = ['E', 'P', 'F1', 'ACC', 'NMI', 'PREC', 'REC', 'ARI']
    results = computeEvaluationMeasures(trueL, predictedL)
    table = [[results[m] for m in MEASURES]]
    return tabulate(table, headers=MEASURES, tablefmt=tabFmt)
