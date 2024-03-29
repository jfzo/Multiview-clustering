import itertools
from scipy.stats import kendalltau
from scipy.io import loadmat
from logging_setup import logger
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, precision_score, recall_score, adjusted_rand_score

import json
import numpy as np
from tabulate import tabulate, tabulate_formats
import numpy as np


def checkExceptionList(A, epweights_A):
    """
    Checks if every point in exception_in_i and cluster label in existLbls_i
    ... have an entry in  epweights_i .
    >> checkExceptionList(self.Pi[i], exception_weights.get(i, {}) )
    """
    existLbls_A = np.unique(A[np.where(A > -1)])
    exception_in_A = np.where(A == -1)[0]
    issues2fix = []
    for pt_i in exception_in_A: # foreach unfused point
        issueFlag = False
        for exsLbl in existLbls_A: # foreach available label
            if not (pt_i, exsLbl) in epweights_A:
                issueFlag = True
        if issueFlag:
            issues2fix.append(pt_i)
    return issues2fix


def repairCorrelative(A):
    arr_A = np.array(A)
    current_ids = np.unique(arr_A)
    L = len(current_ids)
    correct_ids = np.arange(L)
    for i,j in zip(current_ids, correct_ids):
        if i != j:
            arr_A[np.where(arr_A == i)[0]] = j
    return arr_A


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
    if not check_correlative_cluster_labels(S1):
        logger.debug("reparing non correlative label partitions {0}".format(S1))
        S1 = repairCorrelative(S1)
    if not check_correlative_cluster_labels(S2):
        logger.debug("reparing non correlative label partitions {0}".format(S2))
        S2 = repairCorrelative(S2)
    
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
    Each objects is assigned to a cluster id in [0, Kmax[.
    """
    rnd_c = [np.random.randint(0, Kmax) for i in range(n)]
    
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
    return available_labels[np.argmax([agreement_measure(P, set(np.where(B == l)[0]) ) for l in available_labels] )]
        

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
    Phi_A_pi = max_agreement_partition(pi_, A) # max agreement partition in view A.
    
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

def complexity_rank_behavior(results):
    output = {}
    for M in results.keys():
        output[M] = {}
        for DS in results[M].keys():
            if 'aveK_ranks' in results[M][DS]:
                K = results[M][DS]['consensus']["K"]
                # results[M][DS]['aveK_ranks'] is a list of str
                rank_lists = [r_str.split(",") for r_str in results[M][DS]['aveK_ranks']]
                rank_lists_str = str([r_str for r_str in results[M][DS]['aveK_ranks']]).replace("[","").replace("]","").replace("',","' ").replace("'","")
                taus = [kendalltau(a,b)[0] for a,b in itertools.combinations(rank_lists, 2)]
                ave_t, sd_t = np.mean(taus), np.std(taus)
                output[M][DS] = {'k':K, 'ave_tau':ave_t, 'sd_tau':sd_t, 'ranks_str':rank_lists_str}
    return output

def flat_relative_performance_dict(results, report_relative = False):
    """
    Generates a dictionary with each method and view along with the best results for that method over that view.
    The consensus keys are the proposals. The other ones denote the original views clustered by kmeans.
    :param results: Dict with the results
    :return:
    """
    #ov_max_purity = -float("inf")
    #ov_min_entropy = float("inf")
    output = {}
    for M in results.keys():
        for DS in results[M].keys():
            ov_max_purity = -float("inf")
            ov_min_entropy = float("inf")
            ov_max_f1 = -float("inf")
            ov_max_acc = -float("inf")
            ov_max_nmi = -float("inf")
            ov_max_prec = -float("inf")
            ov_max_rec = -float("inf")
            ov_max_ari = -float("inf")
            for V in results[M][DS].keys():
                if V == 'aveK_ranks':  # average complexity ranking Info (avoid its parsing)
                    continue
                entry_name = "{0}_{1}_{2}_k_{3}".format(M, V, DS, DS.split(":")[1])
                output[entry_name] = {}
                output[entry_name]['entropy'] = np.min( results[M][DS][V]['entropy'] )
                output[entry_name]['purity'] = np.max(results[M][DS][V]['purity'])
                output[entry_name]['f1'] = np.max(results[M][DS][V]['f1'])
                output[entry_name]['accuracy'] = np.max(results[M][DS][V]['accuracy'])
                output[entry_name]['nmi'] = np.max(results[M][DS][V]['nmi'])
                output[entry_name]['precision'] = np.max(results[M][DS][V]['precision'])
                output[entry_name]['recall'] = np.max(results[M][DS][V]['recall'])
                output[entry_name]['ari'] = np.max(results[M][DS][V]['ari'])

                ov_min_entropy = output[entry_name]['entropy'] if output[entry_name]['entropy'] < ov_min_entropy else ov_min_entropy
                ov_max_purity = output[entry_name]['purity'] if output[entry_name]['purity'] > ov_max_purity else ov_max_purity
                ov_max_f1 = output[entry_name]['f1'] if output[entry_name]['f1'] > ov_max_f1 else ov_max_f1
                ov_max_acc = output[entry_name]['accuracy'] if output[entry_name]['accuracy'] > ov_max_acc else ov_max_acc
                ov_max_nmi = output[entry_name]['nmi'] if output[entry_name]['nmi'] > ov_max_nmi else ov_max_nmi
                ov_max_prec = output[entry_name]['precision'] if output[entry_name]['precision'] > ov_max_prec else ov_max_prec
                ov_max_rec = output[entry_name]['recall'] if output[entry_name]['recall'] > ov_max_rec else ov_max_rec
                ov_max_ari = output[entry_name]['ari'] if output[entry_name]['ari'] > ov_max_ari else ov_max_ari


            # relative Perf. for all these views
            if report_relative:
                for V in results[M][DS].keys():
                    if V == 'aveK_ranks': # average complexity ranking Info (avoid its parsing)
                        continue
                    entry_name = "{0}_{1}_{2}_k_{3}".format(M, V, DS, DS.split(":")[1])
                    output[entry_name]['entropy'] = output[entry_name]['entropy'] / ov_min_entropy
                    output[entry_name]['purity'] = ov_max_purity / output[entry_name]['purity']
                    output[entry_name]['f1'] = ov_max_f1 / output[entry_name]['f1']
                    output[entry_name]['accuracy'] = ov_max_acc / output[entry_name]['accuracy']
                    output[entry_name]['nmi'] = ov_max_nmi / output[entry_name]['nmi']
                    output[entry_name]['precision'] = ov_max_prec / output[entry_name]['precision']
                    output[entry_name]['recall'] = ov_max_rec / output[entry_name]['recall']
                    output[entry_name]['ari'] = ov_max_ari / output[entry_name]['ari']

    return output


def tab_from_flat(flat_results, measure):
    """
    Generates a lists of lists ready to be displayed by the tabulate package.
    :param flat_results: flat dictionary built by 'flat_performance_dict' function.
    :return: A tuple with the column names and a list of lists ready to be displayed with tabulate package.
    """
    cols = ["k", "dataset"]
    methods = {}
    for k in flat_results:
        columns = k.split("_")
        if not columns[1] in cols:
            cols.append(columns[1])

    tb_results = {}
    for k in flat_results:
        columns = k.split("_")
        # method columns[0]
        # view columns[1]
        # dataset columns[2]
        # columns[4] k-value
        if not columns[0] in tb_results:
            tb_results[columns[0]] = {}
        if not columns[2] in tb_results[columns[0]]:
            tb_results[columns[0]][columns[2]] = [None for i in range(len(cols))]
            tb_results[columns[0]][columns[2]][0] = columns[4]
            tb_results[columns[0]][columns[2]][1] = columns[2]

        i = cols.index(columns[1])
        tb_results[columns[0]][columns[2]][i] = flat_results[k][measure]

    # print(flat_results)
    #print(cols)
    #print(tb_results)
    cols.insert(0, 'Method')
    tb_data = [[M] + perf for M, P in tb_results.items() for d, perf in P.items()]
    return cols, tb_data

def summary_tables_from_json(resultsFile, path = ".", writeSD=True, tableFmt = 'github'):
    #print(random_partition(5, 10))

    path = path.replace("\\", "/")
    #R = json.load(open("RES_Jul282020.062321_29580secs.json"))
    R = json.load(open("{0}/{1}".format(path, resultsFile)) )

    """
       ['fancy_grid', 'github', 'grid', 'html', 'jira', 'latex', 'latex_booktabs', 'latex_raw', 
       'mediawiki', 'moinmoin', 'orgtbl', 'pipe', 'plain', 'presto', 'pretty', 'psql', 'rst', 
       'simple', 'textile', 'tsv', 'youtrack']
    """
    measures = ['entropy', 'purity', 'f1', 'accuracy', 'nmi', 'precision', 'recall', 'ari']

    for met in R:
        for DS in R[met]:
            rows = []
            for V in R[met][DS]:
                if V == 'aveK_ranks':
                    continue
                colHeader = [V]
                colHeader.extend(measures)
                row = [V]
                for M in measures:
                    aveM = np.mean(R[met][DS][V][M])
                    sdM = np.std(R[met][DS][V][M])
                    if writeSD:
                        row.append("{0:.3f} ({1:.3f})".format(aveM, sdM))
                    else:
                        row.append("{0:.3f}".format(aveM))
                rows.append(row)
            print("**{0}**".format(DS))
            print(tabulate(rows, headers=colHeader, tablefmt=tableFmt))

def complete_tables_from_json(resultsFile, path = ".",tableFmt = 'github', outputFile=None):
    path = path.replace("\\", "/")
    R = json.load(open("{0}/{1}".format(path, resultsFile)) )

    """
       ['fancy_grid', 'github', 'grid', 'html', 'jira', 'latex', 'latex_booktabs', 'latex_raw', 
       'mediawiki', 'moinmoin', 'orgtbl', 'pipe', 'plain', 'presto', 'pretty', 'psql', 'rst', 
       'simple', 'textile', 'tsv', 'youtrack']
    """
    measures = ['entropy', 'purity', 'f1', 'accuracy', 'nmi', 'precision', 'recall', 'ari']

    rows = []
    colHeader = []

    for met in R:
        for DS in R[met]:
            for V in R[met][DS]:
                if V in ['aveK_ranks','consensus-internals']:
                    continue
                if len(colHeader) == 0:
                    colHeader = ['DS', 'View', 'run']
                    colHeader.extend(measures)
                nruns = len(R[met][DS][V][measures[0]]) # sometimes this value varies due to absent solutions
                for run in range(nruns):
                    row = [DS, V]
                    row.append(run + 1)
                    for M in measures:
                        row.append("{0:.3f}".format(R[met][DS][V][M][run]))
                    rows.append(row)
    tabularData = tabulate(rows, headers=colHeader, tablefmt=tableFmt)
    if outputFile is None:
        print(tabularData)
    else:
        with open(outputFile, "w") as fp:
            fp.write(tabularData)


def sparseMatFromCluto(inputfile, sparseFmt = False):
    from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
    import numpy as np

    in_fm = open(inputfile)
    N, D, _ = map(int, in_fm.readline().strip().split())  # Number of instances, Number of dimensions and NNZ

    X = lil_matrix((N, D))
    ln_no = 0
    for L in in_fm:
        inst_fields = L.strip().split(" ")
        for i in range(0, len(inst_fields), 2):
            feat = int(inst_fields[i]) - 1  # cluto starts column indexes at 1
            feat_val = float(inst_fields[i + 1])
            X[ln_no, feat] = feat_val

        ln_no += 1

    in_fm.close()

    assert (ln_no == N)
    if sparseFmt:
        return csr_matrix(X)
    #np.savetxt(csv_fname, X.todense(), delimiter=" ")
    return X.todense()

# usado para procesar json que contiene medidas internas de evaluacion.
def process_internal_evaluations(R):
    """
    R is a json object previously created from an input file.
    :param R:
    :return:
    """
    flatResults = {}
    for Mt in R.keys():
        methodResults = R[Mt]
        flatResults[Mt] = {}
        for DS in methodResults.keys():
            flatResults[Mt][DS] = {}
            for VS in methodResults[DS]['consensus-internals'].keys():
                flatResults[Mt][DS][VS] = {}
                # each view has its relative in methodResults[DS]
                aveSil = np.mean(methodResults[DS]['consensus-internals'][VS]['silhouette'])
                clustSil = methodResults[DS][VS]['silhouette']  # obtained from the clusterer performance
                aveCal = np.mean(methodResults[DS]['consensus-internals'][VS]['calinski'])
                clustCal = methodResults[DS][VS]['calinski']  # obtained from the clusterer performance
                aveDvs = np.mean(methodResults[DS]['consensus-internals'][VS]['davies'])
                clustDvs = methodResults[DS][VS]['davies'] # obtained from the clusterer performance
                flatResults[Mt][DS][VS]['silhouette'] = (clustSil, aveSil)
                flatResults[Mt][DS][VS]['calinski'] = (clustCal, aveCal)
                flatResults[Mt][DS][VS]['davies'] = (clustDvs, aveDvs)
    for Mt in flatResults:
        alltbldata = {}
        for ds in flatResults[Mt]:
            tbldata = []
            for vw in flatResults[Mt][ds]:
                tbldata.append((vw, flatResults[Mt][ds][vw]['silhouette'][0][0],
                                flatResults[Mt][ds][vw]['silhouette'][1], flatResults[Mt][ds][vw]['calinski'][0][0],
                                flatResults[Mt][ds][vw]['calinski'][1], flatResults[Mt][ds][vw]['davies'][0][0],
                                flatResults[Mt][ds][vw]['davies'][1]))
            alltbldata[ds] = tbldata

        for DS in alltbldata:
            print("****************" , DS)
            print(tabulate(alltbldata[DS],
                           headers=['view', 'ct-val', 'cns-ave-val', 'ct-val', 'cns-ave-val', 'ct-val', 'cns-ave-val'],
                           tablefmt='latex'))

    return flatResults


def process_benchmark_results():
    cHdr = ('dataset', 'method', 'entropy', 'purity', 'f1', 'accuracy', 'nmi', 'precision', 'recall', 'ari')
    MATLABRESINFO = {'bbc4':'/home/juan/Documentos/ensembles-matlab/bbc4_ensemble_results.mat',
                     'handwritten':'/home/juan/Documentos/ensembles-matlab/handwritten_ensemble_results.mat',
                     'caltech-20':'/home/juan/Documentos/ensembles-matlab/caltech-20_ensemble_results.mat',
                     'Nus-Wide':'/home/juan/Documentos/ensembles-matlab/nusWide_ensemble_results.mat'}
    results = []
    for dsNm, dsResPath in MATLABRESINFO.items():
        mDF = loadmat(dsResPath)
        for embMet in ('cspa', 'hgpa', 'mcla'):
            results.append(
                (dsNm, embMet,
                 Entropy(mDF[embMet].reshape((-1,)), mDF['trLbls'].reshape((-1,))),
                 Purity(mDF[embMet].reshape((-1,)), mDF['trLbls'].reshape((-1,))),
                 f1_score(mDF['trLbls'].reshape((-1,)), mDF[embMet].reshape((-1,)) , average='weighted'),
                 accuracy_score(mDF['trLbls'].reshape((-1,)), mDF[embMet].reshape((-1,))),
                 normalized_mutual_info_score(mDF['trLbls'].reshape((-1,)), mDF[embMet].reshape((-1,)) ),
                 precision_score(mDF['trLbls'].reshape((-1,)), mDF[embMet].reshape((-1,)), average='weighted'),
                 recall_score(mDF['trLbls'].reshape((-1,)), mDF[embMet].reshape((-1,)), average='weighted'),
                 adjusted_rand_score(mDF['trLbls'].reshape((-1,)), mDF[embMet].reshape((-1,)))
                 )
            )

    print(tabulate(results, headers=cHdr, tablefmt='latex'))

def process_negmm_benchmark_results():
    cHdr = ('dataset', 'method', 'entropy', 'purity', 'f1', 'accuracy', 'nmi', 'precision', 'recall', 'ari')
    MATLABRESINFO = {'bbc4':'/home/juan/Documentos/ensembles-matlab/bbc4_negMM_results.mat',
                     'handwritten':'/home/juan/Documentos/ensembles-matlab/handwritten_negMM_results.mat',
                     'caltech-20':'/home/juan/Documentos/ensembles-matlab/caltech-20_negMM_results.mat'}
                     #'Nus-Wide':'/home/juan/Documentos/ensembles-matlab/nusWide_ensemble_results.mat'} # not enouch mem.
    results = []
    embMet = 'NegMM'
    for dsNm, dsResPath in MATLABRESINFO.items():
        mDF = loadmat(dsResPath)
        results.append(
            (dsNm, embMet,
             Entropy(mDF['negMMLabels'].reshape((-1,)), mDF['trLbls'].reshape((-1,))),
             Purity(mDF['negMMLabels'].reshape((-1,)), mDF['trLbls'].reshape((-1,))),
             f1_score(mDF['trLbls'].reshape((-1,)), mDF['negMMLabels'].reshape((-1,)) , average='weighted'),
             accuracy_score(mDF['trLbls'].reshape((-1,)), mDF['negMMLabels'].reshape((-1,))),
             normalized_mutual_info_score(mDF['trLbls'].reshape((-1,)), mDF['negMMLabels'].reshape((-1,)) ),
             precision_score(mDF['trLbls'].reshape((-1,)), mDF['negMMLabels'].reshape((-1,)), average='weighted'),
             recall_score(mDF['trLbls'].reshape((-1,)), mDF['negMMLabels'].reshape((-1,)), average='weighted'),
             adjusted_rand_score(mDF['trLbls'].reshape((-1,)), mDF['negMMLabels'].reshape((-1,)))
             )
        )

    print(tabulate(results, headers=cHdr, tablefmt='latex'))

if __name__ == '__main__':
    #summary_tables_from_json("./json/NCFwR#80-BBC-seg4.json", tableFmt='latex', writeSD=False)
    #summary_tables_from_json("./json/NCFwR#80-Caltech-20.json", tableFmt='latex', writeSD=False)
    #summary_tables_from_json("./json/NCFwR#80-Handwritten.json", tableFmt='latex', writeSD=False)
    #summary_tables_from_json("./json/NCFwR#80-Nus-Wide.json", tableFmt='latex', writeSD=False)
    #summary_tables_from_json("./json/NCFwR#80-Reuters-5.json", tableFmt='latex', writeSD=False)
    summary_tables_from_json("RES_Nov09_internal_measures.json", tableFmt='latex', writeSD=False)

if __name__ == '__main__2':
    import json
    import numpy as np
    from tabulate import tabulate, tabulate_formats
    #print(random_partition(5, 10))
    path = "."
    path = path.replace("\\", "/")
    #R = json.load(open("RES_Jul282020.062321_29580secs.json"))
    R = json.load(open("{0}/{1}".format(path, "outputfile.json")) )
    tableFmt = 'fancy_grid'

    flat_results = flat_relative_performance_dict(R, report_relative=True)
    """
    ['fancy_grid', 'github', 'grid', 'html', 'jira', 'latex', 'latex_booktabs', 'latex_raw', 
    'mediawiki', 'moinmoin', 'orgtbl', 'pipe', 'plain', 'presto', 'pretty', 'psql', 'rst', 
    'simple', 'textile', 'tsv', 'youtrack']
    """
    cols, tb_data = tab_from_flat(flat_results, measure='purity')
    print("PURITY")
    print(tabulate(tb_data, headers=cols, tablefmt=tableFmt))

    cols, tb_data = tab_from_flat(flat_results, measure='entropy')
    print("ENTROPY")
    print(tabulate(tb_data, headers=cols, tablefmt=tableFmt))

    cols, tb_data = tab_from_flat(flat_results, measure='f1')
    print("F1")
    print(tabulate(tb_data, headers=cols, tablefmt=tableFmt))

    cols, tb_data = tab_from_flat(flat_results, measure='accuracy')
    print("ACCURACY")
    print(tabulate(tb_data, headers=cols, tablefmt=tableFmt))

    cols, tb_data = tab_from_flat(flat_results, measure='nmi')
    print("NMI")
    print(tabulate(tb_data, headers=cols, tablefmt=tableFmt))

    cols, tb_data = tab_from_flat(flat_results, measure='precision')
    print("PRECISION")
    print(tabulate(tb_data, headers=cols, tablefmt=tableFmt))

    cols, tb_data = tab_from_flat(flat_results, measure='recall')
    print("RECALL")
    print(tabulate(tb_data, headers=cols, tablefmt=tableFmt))

    cols, tb_data = tab_from_flat(flat_results, measure='ari')
    print("ARI")
    print(tabulate(tb_data, headers=cols, tablefmt=tableFmt))


    comp_rank_b = complexity_rank_behavior(R)
    #print(comp_rank_b)
    flat_comp = [('method',"nrnd",'dataset','k','ave-tau','sd-tau','ranks')]
    for M in comp_rank_b:
        for D in comp_rank_b[M]:
            flat_comp.append((M.split("#")[0],M.split("#")[1],D.split(":")[0],D.split(":")[1],comp_rank_b[M][D]['ave_tau'],comp_rank_b[M][D]['sd_tau'],comp_rank_b[M][D]['ranks_str']))

    #print(flat_comp)
    
    print(tabulate(flat_comp[1:], headers=flat_comp[0], tablefmt=tableFmt))