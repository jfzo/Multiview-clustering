from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
import os
from sklearn import preprocessing
import csv
from tabulate import tabulate as tab
import scipy.sparse
import pickle
from tabulate import tabulate as tab

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random

from scoop import futures

import logging

import fraj_proposal as fraj

import argparse


def random_partition(Kmin, Kmax, n):
    """
    Generates a random assignment of `n` objects into `K` clusters
    `K` will be randomly chosen between `Kmin` `Kmax`
    """
    
    # TODO: Fix this. Kmin is not been used.
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


#######

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
    # epweights_A must contain the tuple due to the former mark of p in A
    return agreement_measure(pi_, set(np.where(A == Phi_A_pi)[0])) * epweights_A[(p, Phi_A_pi)]
    
########################### G.A ########################
def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2


def repairCorrelative(A):
    current_ids = np.unique(A)
    L = len(current_ids)
    correct_ids = np.arange(L)
    for i,j in zip(current_ids, correct_ids):
        if i != j:
            A[np.where(A == i)[0]] = j

def evalMatching(C, A=None, B=None, K_A=None, K_B=None):
    """
    Objective function for the G.A. strategy to merging partitions.
    """
    #K_A = np.unique(A).shape[0] #can be computed outside
    #K_B = np.unique(B).shape[0] #can be computed outside
    repairCorrelative(C)
    K_C = np.unique(C).shape[0]

    K_C_A = compute_solutions_complexity(C, A, K_C, K_A)[2]
    K_C_B = compute_solutions_complexity(C, B, K_C, K_B)[2]
    
    K_A_C = compute_solutions_complexity(A, C, K_A, K_C)[2]
    K_B_C = compute_solutions_complexity(B, C, K_B, K_C)[2]
    
    return K_C_A + K_C_B + K_A_C + K_B_C,


###########
# If the following instructions are created inside the ga_best_merge 
# function, SCOOP fails!
###

creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.Fitness)

tb1 = base.Toolbox()
tb1.register("map", futures.map)
tb1.register("mate", cxTwoPointCopy) # cxTwoPointCopy defined above
#tb1.register("mutate", tools.mutFlipBit, indpb=0.05) # PARAM
#mutUniformInt(individual, low, up, indpb)
#tb1.register("mutate", tools.mutUniformInt, indpb=0.05, low=, up=)
#tb1.register("select", tools.selRoulette)  # PARAM
tb1.register("select", tools.selTournament, tournsize=3)  # PARAM

###############




def ga_find_best_merge(V1, V2, K_V1, K_V2, popsize=300, seed=1):



    #K_V1 = np.unique(V1).shape[0]
    #V2 = np.array([0,0,1,0,0,1,1,2,2,2], dtype=np.int)
    #K_V2 = np.unique(V2).shape[0]

    NCLUSTERS = np.max([K_V1, K_V2])
    NPTS = V1.shape[0]


    #tb1 = base.Toolbox()
    tb1.register("mutate", tools.mutUniformInt, indpb=0.05, low=0, up=NCLUSTERS)
    tb1.register("attr_item", random.randint, 0, NCLUSTERS) # each gene corresponds to a chr
    tb1.register("individual", tools.initRepeat, creator.Individual, tb1.attr_item, NPTS)
    tb1.register("population", tools.initRepeat, list, tb1.individual)
    tb1.register("map", futures.map) # enabling scoop
    
    tb1.register("evaluate", evalMatching, A=V1, B=V2, K_A=K_V1, K_B=K_V2)


    # running the genetic algorithm
    random.seed(seed)
    
    pop = tb1.population(n=popsize)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # eaSimple(population, toolbox, cxpb, mutpb, ngen[, stats, halloffame, verbose])
    # cxpb – The probability that an offspring is produced by crossover.
    """
    Crossover is performed on two parents to form two new offspring. 
    The GA has a crossover probability that determines if crossover will happen. 
    """
    # mutpb – The probability that an offspring is produced by mutation.
    algorithms.eaSimple(pop, tb1, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats,
                        halloffame=hof)
    # eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen[, stats, halloffame, verbose])
    # mu – The number of individuals to select for the next generation.
    # lambda_ – The number of children to produce at each generation.
    best_ind = np.array(tools.selBest(pop, 1)[0])

    return best_ind, stats

def ga_merge(i,j,Pi,K, popsize):

    new_partitioning, _ = ga_find_best_merge(Pi[i], Pi[j], K[i], K[j], popsize=popsize)

    return new_partitioning, {}

def merge(i,j,Pi,K, exception_weights, alpha=0.7):
    """
    i,j : pair of views to merge
    Pi : Collection with the partitionings (1D vector for each view)
    K : Collection with the number of clusters per view.  
    
    Returns:
    marked_point_exceptions: Dict with tuples (point_id-cluster_id) as keys and weights as values.
    alpha: weight of the historic membership accumulated. The weight of the current membership is (1-alpha)
    Note
    ====
    Label -1 identifies points marked as exceptions (is not a cluster) so we take it out.
    """
    labels = list((set(np.unique(Pi[i])) |set(np.unique(Pi[j]))  ) - {-1})
    
    r1,_,_ = compute_solutions_complexity(Pi[i], Pi[j], K[i], K[j], labels=labels)
    r2,_,_ = compute_solutions_complexity(Pi[j], Pi[i], K[j], K[i], labels=labels)
    #print("r1:",r1)
    #print("r2:",r2)
    #[(x,Pi[i][x],Pi[j][x]) for x in (set(e1.keys()) | set(e2.keys()))-(set(e1.keys()) & set(e2.keys()))]

    merge_ops = dict([(u,set()) for u in range(len(r1))])
    for u in range(len(r1)):
        #print("MERGE(%d, %d)"%(u, r1[u]))
        merge_ops[u].add(r1[u])

    for v in range(len(r2)):
        #print("MERGE(%d, %d)"%(v, r2[v]))
        merge_ops[r2[v]].add(v)
    #print(merge_ops)
    
    cache = {} #employed for storing the updated versions of the partitions not merged.
    #new_partitioning = np.zeros(Pi[i].shape, dtype=np.int32)
    new_partitioning = np.repeat(-1, Pi[i].shape[0]) # By default each pt is marked.
    cluster_id = 0 # Cluster ids start from 0
    overall_candidates = set()
    
    for e,s in merge_ops.items():
        # candidate points to merge initially
        set_vie = set(np.where(Pi[i] == e)[0])
        # CREATE A NEW CLUSTER
        c = set()
        for p in s: # p is an integer denoting a cluster in view j 
            # add to c only candidates & p
            set_vjp = None
            if p in cache:
                set_vjp = cache[p] # not a copy
            else:
                set_vjp = set(np.where(Pi[j] == p)[0])
            c |= (set_vie & set_vjp) # adding intersection to the new cluster
            a = set_vie - set_vjp # the remaining items to merge for next p's in s
            # Update set_vjp to (set_vjp - set_vie ) to use in further (e,s)            
            cache[p] = set_vjp - set_vie 
            set_vie = set(a)

        # when there are no common data points, the merge generates only exceptions.
        if len(c) == 0:
            logger.error("ERROR empty cluster when merging views i={} j={}: No common \
points between partition {} and {}".format(i,j,e,s))
            logger.warning("|view {} partition {}| : {}".format(i,e,len(set_vie)))
            for pt in s:
                logger.warning("|view {} partition {}| : {}({})".format(j,pt,len(cache[pt]),len(set(np.where(Pi[j] == pt)[0])) ) )
            # maybe this issue shoud throw an exception since the result is no longer valid (Bad source clustering).
            assert len(c) > 0
            
        new_partitioning[np.array(list(c), dtype=np.int32)] = cluster_id
        cluster_id += 1
        # the remaining points in View i that couldn't be merged are marked as potential exceptions.
        overall_candidates |= set_vie
        # end of the creation of the new partition
        
    # Additionally, Points that couldn't be included in any merge must also be marked.
    for p,set_vjp in cache.items():
        overall_candidates |= set_vjp 
    # end if Merging step.

    # computing the weights of marked points
    marked_points_weights = {}
    for p in overall_candidates:        
        available_labels = np.unique(new_partitioning[np.where(new_partitioning > -1)[0]])
        epweights_A = exception_weights.get(i, {}) # original source views dont have exceptions
        epweights_B = exception_weights.get(j, {})
        w_a = [omega(set(np.where(new_partitioning == l)[0]), p, Pi[i], epweights_A) for l in available_labels]
        w_b = [omega(set(np.where(new_partitioning == l)[0]), p, Pi[j], epweights_B) for l in available_labels]
        
        for l_ix in range(len(available_labels)): # generally, the l_ix must match the label id.
            W_pl = 0.5*w_a[l_ix] + 0.5*w_b[l_ix]
            marked_points_weights[(p, available_labels[l_ix])] = W_pl
        
    return new_partitioning, marked_points_weights

def ga_proposal(result, popsize):
    """
    Version that employes a GA optimization strategy for merging partitions
    """
    logger.info("Executing GA-merging proposal")

    REFERENCE_VALUE = float('Inf') # this value is used for minimum-distance merging
    # capital Pi: list of partitionings.
    # A partitioning is a set of integers (data point id)
    Pi = [x for name,x in result.items()]
    m = len(Pi) # nr. of views
    logger.debug("initial number of views: {}".format(m))
    K = [np.unique(k).shape[0] for k in Pi]

    ######
    # Distance matrix computation
    #
    Aff = np.repeat(REFERENCE_VALUE, m**2).reshape(m, m)

    for i in range(m-1):
        for j in range(i+1,m):
            _,_,val1 = compute_solutions_complexity(Pi[i], Pi[j], K[i], K[j])
            _,_,val2 = compute_solutions_complexity(Pi[j], Pi[i], K[j], K[i])
            #if(len(set(e1.keys()) & set(e2.keys())) != len(set(e1.keys()) | set(e2.keys())) ):
            #    print("C1:%d C2:%d"%(i,j))
            #else:
            #    print("*C1:%d C2:%d"%(i,j))
            Aff[i,j] = val1 + val2
            Aff[j,i] = Aff[i,j]
    #print(Aff)

    processed_views = set() # stores the partitionings already merged so they are not re-used
    #marked_pts = {} # dictionary with point ids (integer) as keys and list of tuples (cluster-id; weight)
    #
    # the following must be repeated until no more partitionings can be merged.
    # It seems that the condition is that the min value of the distance matrix is Inf.
    exception_weights = {} # dict with View_id as key and a dict ((point_id,cluster_id)->weight) as value.
    n_its = 0
    last_created_view = -1
    while( not np.isinf(np.min(Aff)) ):
        #print("merge views %d and %d"%(mindist_partitions, row_min_dist) )
        # Picking current views having minimum complexity:
        optimal_col_per_row = np.argmin(Aff, axis=1) # for each row, which column presents the lowest distance
        optimal_val_per_row = np.min(Aff, axis=1)
        optimal_row = np.argmin(optimal_val_per_row) # row whose min distance is the global min.
        optimal_col = optimal_col_per_row[optimal_row] # combining the previous commands, obtain the least distant views to merge.

        newclustering, exceptions = ga_merge(optimal_row, optimal_col, Pi, K, popsize)
        #step_memberships.append(memberships_)
        K.append(np.unique(newclustering[np.where(newclustering >= 0)[0]]).shape[0]) 
        Pi.append(newclustering.copy())
        newclustering_index = len(Pi)-1
        last_created_view = newclustering_index

        logger.debug("merge views {} and {} --> {} is created.".format(optimal_row, optimal_col, newclustering_index) )

        # TODO: add exceptions to the overall list
        #for e,_ in exceptions:
        #    if e not in marked_pts:
        #        marked_pts[e] = []
        #    marked_pts[e].append((newclustering_index, 1/merging_dist))

        # set the rows/columns of merged partitions as processed!
        processed_views.add(optimal_row)
        processed_views.add(optimal_col)

        # Update the distance matrix: Set the merged partitionings distances to infinity so they
        # are not employed in posterior iterations ~ Removing these partitions from the set of candidates
        Aff[optimal_col,:] = np.repeat(REFERENCE_VALUE, m)
        Aff[:,optimal_col] = np.repeat(REFERENCE_VALUE, m)
        Aff[optimal_row,:] = np.repeat(REFERENCE_VALUE, m)
        Aff[:,optimal_row] = np.repeat(REFERENCE_VALUE, m)

        # Include the new partitioning into the set of views:
        #    * A row and column are added to represent the new clustering.
        #    * Distances between this new partitioning and the valid ones are computed.
        Aff = np.vstack(( 
            np.hstack((
                Aff, np.repeat(REFERENCE_VALUE, Aff.shape[0]).reshape(Aff.shape[0],1) )), 
            np.repeat(REFERENCE_VALUE, Aff.shape[0]+1) ))

        #print("1st partition:",np.unique(Pi[newclustering_index]))
        # Update the Kolmogorov complexities
        for j in range(len(Pi)):
            if j in processed_views or j == newclustering_index:
                continue
            _,_,val1 = compute_solutions_complexity(Pi[newclustering_index], Pi[j], K[newclustering_index], K[j])
            _,_,val2 = compute_solutions_complexity(Pi[j], Pi[newclustering_index], K[j], K[newclustering_index])

            Aff[newclustering_index,j] = val1 + val2
            Aff[j,newclustering_index] = Aff[newclustering_index,j]

        #update the m value
        m = len(Pi)

        n_its += 1
        if n_its > 50:
            logger.warning("Nr. of max iterations reached. Halting!")
            break
   
    return Pi[last_created_view]



def new_proposal(result):
    """
    result: Partitionings as integer vectors. One vector for each view.
    """
    logger.info("Executing NCI proposal")
    REFERENCE_VALUE = float('Inf') # this value is used for minimum-distance merging
        
    # capital Pi: list of partitionings.
    # A partitioning is a set of integers (data point id)
    Pi = [x for name,x in result.items()]
    m = len(Pi) # nr. of views
    
    logger.debug("initial number of views:".format(m) )
    K = [np.unique(k).shape[0] for k in Pi]

    n = len(Pi[0])
    N_rnd = 50
    kmax = np.max(K)
    rnd_p = [random_partition(None, kmax, n) for i in range(N_rnd)]    
    # ranking
    ave_complexity = [np.mean([compute_solutions_complexity(Pi[j], rnd_p[i], K[j], kmax)[2] for i in range(N_rnd)]) for j in range(m)]
    # if ave_complexity[i] > ave_complexity[j] => view_i has more information thant view_j, thus it is ranked higher
    
    ######
    # Distance matrix computation
    #
    Aff = np.repeat(REFERENCE_VALUE, m**2).reshape(m, m)

    for i in range(m-1):
        for j in range(i+1,m):
            _,_,val1 = compute_solutions_complexity(Pi[i], Pi[j], K[i], K[j])
            _,_,val2 = compute_solutions_complexity(Pi[j], Pi[i], K[j], K[i])
            #if(len(set(e1.keys()) & set(e2.keys())) != len(set(e1.keys()) | set(e2.keys())) ):
            #    print("C1:%d C2:%d"%(i,j))
            #else:
            #    print("*C1:%d C2:%d"%(i,j))
            Aff[i,j] = val1 + val2
            Aff[j,i] = Aff[i,j]
    #print(Aff)

    processed_views = set() # stores the partitionings already merged so they are not re-used
    #marked_pts = {} # dictionary with point ids (integer) as keys and list of tuples (cluster-id; weight)
    #
    # the following must be repeated until no more partitionings can be merged.
    # It seems that the condition is that the min value of the distance matrix is Inf.
    exception_weights = {} # dict with View_id as key and a dict ((point_id,cluster_id)->weight) as value.
    n_its = 0
    last_created_view = -1
    while( not np.isinf(np.min(Aff)) ):
        #mindist_part_for_rows = np.argmin(Aff, axis=1) # for each row, which column presents the lowest distance
        #mindist_partitions = np.argmin(np.min(Aff, axis=1)) # row whose min distance is the global min.
        #row_min_dist = mindist_part_for_rows[mindist_partitions] # combining the previous commands, obtain the least distant views to merge.

        #print("merge views %d and %d"%(mindist_partitions, row_min_dist) )
        #merging_dist = np.min(np.min(Aff, axis=1))
        #newclustering, exceptions, memberships_ = merge(mindist_partitions, row_min_dist, Pi, K, step_memberships, alpha=0.1)

        optimal_col_per_row = np.argmin(Aff, axis=1) # for each row, which column presents the lowest distance
        optimal_val_per_row = np.min(Aff, axis=1)
        optimal_row = np.argmin(optimal_val_per_row) # row whose min distance is the global min.
        optimal_col = optimal_col_per_row[optimal_row] # combining the previous commands, obtain the least distant views to merge.

        #merging_dist = np.min(np.min(Aff, axis=1))

        #print(Aff)

        # original version had only the following line
        #newclustering, exceptions = merge(optimal_row, optimal_col, Pi, K, exception_weights, alpha=0.1)
        # now the ranked version
        if ave_complexity[optimal_row] > ave_complexity[optimal_col]:
            newclustering, exceptions = merge(optimal_row, optimal_col, Pi, K, exception_weights, alpha=0.1)
        else:
            newclustering, exceptions = merge(optimal_col, optimal_row, Pi, K, exception_weights, alpha=0.1)
        
        #step_memberships.append(memberships_)
        K.append(np.unique(newclustering[np.where(newclustering >= 0)[0]]).shape[0]) 
        Pi.append(newclustering.copy())
        newclustering_index = len(Pi)-1
        last_created_view = newclustering_index
        exception_weights[newclustering_index] = exceptions
        
        # rank for the new view
        new_complexity = np.mean([compute_solutions_complexity(Pi[newclustering_index], rnd_p[i], K[newclustering_index], kmax)[2] for i in range(N_rnd)])
        ave_complexity.append(new_complexity)
        logger.debug("Newly created view has reference average complexity {}".format(new_complexity))
        
        logger.debug("merge views {} and {} --> {} is created.".format(optimal_row, optimal_col, newclustering_index) )

        # TODO: add exceptions to the overall list
        #for e,_ in exceptions:
        #    if e not in marked_pts:
        #        marked_pts[e] = []
        #    marked_pts[e].append((newclustering_index, 1/merging_dist))

        # set the rows/columns of merged partitions as processed!
        processed_views.add(optimal_row)
        processed_views.add(optimal_col)

        # Update the distance matrix: Set the merged partitionings distances to infinity so they
        # are not employed in posterior iterations ~ Removing these partitions from the set of candidates
        Aff[optimal_col,:] = np.repeat(REFERENCE_VALUE, m)
        Aff[:,optimal_col] = np.repeat(REFERENCE_VALUE, m)
        Aff[optimal_row,:] = np.repeat(REFERENCE_VALUE, m)
        Aff[:,optimal_row] = np.repeat(REFERENCE_VALUE, m)

        # Include the new partitioning into the set of views:
        #    * A row and column are added to represent the new clustering.
        #    * Distances between this new partitioning and the valid ones are computed.
        Aff = np.vstack(( 
            np.hstack((
                Aff, np.repeat(REFERENCE_VALUE, Aff.shape[0]).reshape(Aff.shape[0],1) )), 
            np.repeat(REFERENCE_VALUE, Aff.shape[0]+1) ))

        #print("1st partition:",np.unique(Pi[newclustering_index]))
        # Update the Kolmogorov complexities
        for j in range(len(Pi)):
            if j in processed_views or j == newclustering_index:
                continue
            _,_,val1 = compute_solutions_complexity(Pi[newclustering_index], Pi[j], K[newclustering_index], K[j])
            _,_,val2 = compute_solutions_complexity(Pi[j], Pi[newclustering_index], K[j], K[newclustering_index])

            Aff[newclustering_index,j] = val1 + val2
            Aff[j,newclustering_index] = Aff[newclustering_index,j]

        #update the m value
        m = len(Pi)

        n_its += 1
        if n_its > 50:
            logger.error("Nr. of max iterations reached. Halting!")
            break

    # solving the exceptions
    
    logger.debug("Solving the exceptions!")

    max_weight_cluster = {}
    # Since the exceptions are carried, only consider the last view weights
    #print("Solving the exceptions...\n ##################################")
    #print(exceptions[last_created_view],"\n ###############################")
    for ((p_id,c_id), w) in exception_weights[last_created_view].items():
        assert Pi[last_created_view][p_id] == -1 # len(Pi)-1 in line 285
        
        if not p_id in max_weight_cluster:
            max_weight_cluster[p_id] = (c_id, w)
        elif max_weight_cluster[p_id][1] < w:
            max_weight_cluster[p_id] = (c_id, w)
        
    for (p_id, (c,_)) in  max_weight_cluster.items():
        Pi[last_created_view][p_id] = c
    
    return Pi[last_created_view]



def cluster_purity(r, labels_pred, labels_true):
    r_items = np.where(labels_pred == r)[0]
    n_r = len(r_items)
    
    unique_items_lbls, lbls_cnts = np.unique(labels_true[r_items], return_counts=True)
    return (1/n_r)*np.max(lbls_cnts)

def Purity(labels_pred, labels_true):
    L_pred, cnt_pred= np.unique(labels_pred, return_counts=True)
    n = len(labels_pred)
    sum_E = 0
    for r in range(len(L_pred)):
        cp_r = cluster_purity(L_pred[r], labels_pred, labels_true)
        n_r = cnt_pred[r]
        sum_E += (n_r/n)*cp_r
    return sum_E
    
def cluster_entropy(r, labels_pred, labels_true):
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
    L_pred, cnt_pred= np.unique(labels_pred, return_counts=True)
    n = len(labels_pred)
    sum_E = 0
    for r in range(len(L_pred)):
        ce_r = cluster_entropy(L_pred[r], labels_pred, labels_true)
        n_r = cnt_pred[r]
        sum_E += (n_r/n)*ce_r
    return sum_E



    
def run_experimentation(clusters_rng, METHOD = None, POPSIZE=100, NRUNS=10, seed=1, dataset_dir = ".."+os.sep+"data"): # set data directory with the 2nd param.

    datasets = [
        {"lda":"20Newsgroup?20ng_4groups_lda.npz", 
         "skipgram":"20Newsgroup?20ng_4groups_doc2vec.npz",
         "tfidf":"20Newsgroup?20ng-scikit-mat-tfidf.npz",
         "labels":"20Newsgroup?20ng_4groups_labels.csv",
         "dataset":"20Newsgroup"},        
        {"lda":"bbcsport?fulltext?bbcsport-textdata-mat-lda.npz", 
         "skipgram":"bbcsport?fulltext?bbcsport-textdata-mat-skipgram.npz",
         "tfidf":"bbcsport?fulltext?bbcsport-textdata-mat-tfidf.npz",
         "labels":"bbcsport?fulltext?bbcsport-textdata-labels.csv",
         "dataset":"BBCSport"},
        {"lda":"reuters-r8?r8-test-mat-lda.npz",
         "skipgram":"reuters-r8?r8-test-mat-skipgram.npz",
         "tfidf":"reuters-r8?r8-test-mat-tfidf.npz",
         "labels":"reuters-r8?r8-test-labels.txt",
         "dataset":"Reuters-R8"},
        {"lda":"WebKB?webkb-textdata-mat-lda.npz",
         "skipgram":"WebKB?webkb-textdata-mat-skipgram.npz",
         "tfidf":"WebKB?webkb-textdata-mat-tfidf.npz",
         "labels":"WebKB?webkb-textdata-labels.csv",
         "dataset":"WebKB"}
    ]
    
    np.random.seed(seed)
    
    entropy_log = dict()
    for NCLUSTERS in clusters_rng:
        logger.info("Experiments with {} clusters".format(NCLUSTERS))
        
        for ds in datasets:
            run = 0
            while run < NRUNS:
                random_state = np.random.randint(2**16 - 1)
                logger.info("Starting run nr. {} with random state:{}.".format(run, random_state))
                if ds["dataset"] not in entropy_log:
                    entropy_log[ds["dataset"]] = dict()
		            
                with open(dataset_dir+os.sep+ds["labels"].replace("?",os.sep), 'r') as f:
                    reader = csv.reader(f)
                    lst_labels = list(reader)
                    
                lst_labels = [x[0] for x in lst_labels]
                le = preprocessing.LabelEncoder()
                labels = le.fit_transform(lst_labels)
		        
                views = {}

                for viewname in ["tfidf","lda","skipgram"]:
                    if viewname not in entropy_log[ds["dataset"]]:
                        entropy_log[ds["dataset"]][viewname] = dict()

                    if NCLUSTERS not in entropy_log[ds["dataset"]][viewname]:
                        entropy_log[ds["dataset"]][viewname][NCLUSTERS] = dict()
                        entropy_log[ds["dataset"]][viewname][NCLUSTERS]["entropy"] = []
                        entropy_log[ds["dataset"]][viewname][NCLUSTERS]["purity"] = []
		                
		            
		            
                    X = None
                    if viewname == 'tfidf':
                        X = scipy.sparse.load_npz(dataset_dir+os.sep+ds[viewname].replace("?",os.sep))
                    else:
                        X = np.load(dataset_dir+os.sep+ds[viewname].replace("?",os.sep))['arr_0']


		            #for dname, X in {'tfidf':tfidfdata, 'lda':ldadata, 'skipgram':doc2vec_data}.items():
                    km = MiniBatchKMeans(n_clusters=NCLUSTERS,  init='k-means++', n_init=3, init_size=900, batch_size=300, random_state=random_state)
		            #km = KMeans(n_clusters=NCLUSTERS, init='k-means++', max_iter=100, n_init=1,verbose=False, random_state=232343545)

                    km_labels = km.fit_predict(X)
                    entropy_ds_k = Entropy(km_labels, labels)
                    purity_ds_k = Purity(km_labels, labels)
		            
		            
                    entropy_log[ds["dataset"]][viewname][NCLUSTERS]["entropy"].append(entropy_ds_k)
                    entropy_log[ds["dataset"]][viewname][NCLUSTERS]["purity"].append(purity_ds_k)

                    views[viewname] = km_labels

		            #vscore = metrics.v_measure_score(labels, km.labels_)
		            #f1score = metrics.f1_score(labels, km.labels_, average='micro')
		            #nmiscore = metrics.adjusted_mutual_info_score(labels, km.labels_, average_method='arithmetic')
		            #Entropy(labels, labels_T)
                
                #end for viewname
                
		        # execute proposal
		        # catch exception and delete last scores from entropy and purity lists of each viewname record.
                try:
                    assert(METHOD != None)
                    if METHOD == "GA":
                        proposal_result = ga_proposal(views, POPSIZE)
                    elif METHOD == "NCI":
                        proposal_result = new_proposal(views)

                    #proposal_result = new_proposal(views)
                    #proposal_result = ga_proposal(views, POPSIZE)

                    entropy_ds_k = Entropy(proposal_result, labels)
                    purity_ds_k = Purity(proposal_result, labels)
			        
			        
                    if "proposal" not in entropy_log[ds["dataset"]]:
                        entropy_log[ds["dataset"]]["proposal"] = dict()

                    if NCLUSTERS not in entropy_log[ds["dataset"]]["proposal"]:
                        entropy_log[ds["dataset"]]["proposal"][NCLUSTERS] = dict()
                        entropy_log[ds["dataset"]]["proposal"][NCLUSTERS]["entropy"] = []
                        entropy_log[ds["dataset"]]["proposal"][NCLUSTERS]["purity"] = []
			            
                    entropy_log[ds["dataset"]]["proposal"][NCLUSTERS]["entropy"].append(entropy_ds_k)
                    entropy_log[ds["dataset"]]["proposal"][NCLUSTERS]["purity"].append(purity_ds_k)
                except AssertionError as e:
                    logger.error(e)
                    for viewname in ["tfidf","lda","skipgram"]:
                        entropy_log[ds["dataset"]][viewname][NCLUSTERS]["entropy"].pop()
                        entropy_log[ds["dataset"]][viewname][NCLUSTERS]["purity"].pop()
                    continue
                    
                ##############################
                # BASELINE ( Fraj et al. 2019)
                fraj_proposal_result = fraj.ensemble_clustering([views[vname] for vname in ["tfidf","lda","skipgram"]], NCLUSTERS)

                entropy_ds_k = Entropy(fraj_proposal_result, labels)
                purity_ds_k = Purity(fraj_proposal_result, labels)

                if "fraj" not in entropy_log[ds["dataset"]]:
                    entropy_log[ds["dataset"]]["fraj"] = dict()

                if NCLUSTERS not in entropy_log[ds["dataset"]]["fraj"]:
                    entropy_log[ds["dataset"]]["fraj"][NCLUSTERS] = dict()
                    entropy_log[ds["dataset"]]["fraj"][NCLUSTERS]["entropy"] = []
                    entropy_log[ds["dataset"]]["fraj"][NCLUSTERS]["purity"] = []

                entropy_log[ds["dataset"]]["fraj"][NCLUSTERS]["entropy"].append(entropy_ds_k)
                entropy_log[ds["dataset"]]["fraj"][NCLUSTERS]["purity"].append(purity_ds_k)

                
                run += 1
            #end while run
        #end for dataset
    #end for nclusters
    pickle.dump( entropy_log, open( "entropy_log.p", "wb" ) )
    
    
# performance measures

def __results_min_entropy_per_dataset(results, nclusters_rng):

    #results = pickle.load(open(pickle_results_file,'rb'))
    min_entropies = dict()
    for k in nclusters_rng:
        min_entropies[k] = dict()
        for ds in results:
            min_entropies[k][ds] = float('Inf')
            for view in results[ds]:
                # compute minimum for the view and the dataset and the nr of clusters
                minent_k_ds_v = np.min(results[ds][view][k]['entropy'])
                if minent_k_ds_v < min_entropies[k][ds]:
                    min_entropies[k][ds] = minent_k_ds_v
    return min_entropies

def __results_max_purity_per_dataset(results, nclusters_rng):

    max_purity = dict()
    for k in nclusters_rng:
        max_purity[k] = dict()
        for ds in results:
            max_purity[k][ds] = -float('Inf')
            for view in results[ds]:
                # compute minimum for the view and the dataset and the nr of clusters
                maxpur_k_ds_v = np.max(results[ds][view][k]['purity'])
                if maxpur_k_ds_v > max_purity[k][ds]:
                    max_purity[k][ds] = maxpur_k_ds_v
    return max_purity


def results_rel_entropy_per_dataset(pickle_results_file, nclusters_rng):

    results = pickle.load(open(pickle_results_file,'rb'))
    min_entropies = __results_min_entropy_per_dataset(results, nclusters_rng)

    rel_entropies = dict()
    for k in nclusters_rng:
        rel_entropies[k] = dict()
        for ds in results:
            rel_entropies[k][ds] = dict()
            #min_entropies[k][ds] = float('Inf')
            for view in results[ds]:
                # compute minimum for the view and the dataset and the nr of clusters
                minent_k_ds_v = np.min(results[ds][view][k]['entropy'])
                rel_entropies[k][ds][view] = minent_k_ds_v / min_entropies[k][ds]

    return rel_entropies


def results_rel_purity_per_dataset(pickle_results_file, nclusters_rng):

    results = pickle.load(open(pickle_results_file,'rb'))
    max_purities = __results_max_purity_per_dataset(results, nclusters_rng)

    rel_purities = dict()
    for k in nclusters_rng:
        rel_purities[k] = dict()
        for ds in results:
            rel_purities[k][ds] = dict()
            #min_entropies[k][ds] = float('Inf')
            for view in results[ds]:
                # compute maximum for the view and the dataset and the nr of clusters
                maxpur_k_ds_v = np.max(results[ds][view][k]['purity'])
                rel_purities[k][ds][view] = max_purities[k][ds] / maxpur_k_ds_v
                logger.info('REL PURITY: {:.4f} / {:.4f} = {:.4f}'.format(max_purities[k][ds], maxpur_k_ds_v, max_purities[k][ds] / maxpur_k_ds_v))

    return rel_purities

def results_ave_entropy_per_dataset(pickle_results_file, nclusters_rng):
    rel_entropies = results_rel_entropy_per_dataset(pickle_results_file, nclusters_rng)
    ave_entropies = dict()
    #rel_entropies[k][ds]
    for k in nclusters_rng:
        n_datasets = len(rel_entropies[k])
        ave_entropies[k] = dict()
        for v in ['tfidf','lda','skipgram','proposal','fraj']:
            #rel_entropies[k][?][view]
            sum_rel = 0
            for ds in rel_entropies[k]:
                sum_rel += rel_entropies[k][ds][v]
            ave_entropies[k][v] = sum_rel / n_datasets
    return ave_entropies

def results_ave_purity_per_dataset(pickle_results_file, nclusters_rng):
    rel_purities = results_rel_purity_per_dataset(pickle_results_file, nclusters_rng)
    ave_puritites = dict()
    #rel_entropies[k][ds]
    for k in nclusters_rng:
        n_datasets = len(rel_purities[k])
        ave_puritites[k] = dict()
        for v in ['tfidf','lda','skipgram','proposal', 'fraj']:
            #rel_entropies[k][?][view]
            sum_rel = 0
            for ds in rel_purities[k]:
                sum_rel += rel_purities[k][ds][v]
            ave_puritites[k][v] = sum_rel / n_datasets
    return ave_puritites


def print_ave_entropy_results(pickle_results_file, nclusters_rng):
    ave_entropies = results_ave_entropy_per_dataset(pickle_results_file, nclusters_rng)
    #print(tab(exp_results, headers=["dataset","view","f1","nmi","vscore"], floatfmt=".3f", tablefmt="latex") )#"fancy_grid") )
    tab_data = []

    header = ["k"]
	
    views = list(ave_entropies[list(ave_entropies.keys())[0]])

    views.sort()


    for view in views:
        header.append(view)

    for k in nclusters_rng:
        view_row = [k]
        for view in views:
            #logger.debug("*******")
            #logger.debug("view: %s", view)
            view_row.append(ave_entropies[k][view])
        tab_data.append(view_row.copy())

    return tab_data, header




def print_ave_purity_results(pickle_results_file, nclusters_rng):
    ave_purities = results_ave_purity_per_dataset(pickle_results_file, nclusters_rng)
    #print(tab(exp_results, headers=["dataset","view","f1","nmi","vscore"], floatfmt=".3f", tablefmt="latex") )#"fancy_grid") )
    tab_data = []

    header = ["k"]
    views = list(ave_purities[list(ave_purities.keys())[0]])    

    views.sort()

    for view in views:
        header.append(view)

    for k in nclusters_rng:
        view_row = [k]
        for view in views:
            view_row.append(ave_purities[k][view])
        tab_data.append(view_row.copy())

    return tab_data, header


def print_rel_entropy_results(pickle_results_file, nclusters_rng):
    rel_entropies = results_rel_entropy_per_dataset(pickle_results_file, nclusters_rng)
    tab_data = []

    views = list(rel_entropies[list(rel_entropies.keys())[0]]['20Newsgroup'])

    views.sort()

    for k in nclusters_rng:
        for ds in rel_entropies[k]:
            view_row = [k, ds]
            for view in views:
                view_row.append(rel_entropies[k][ds][view])
            tab_data.append(view_row.copy())
    header = ['k', 'dataset']
    header.extend(views)
    return tab_data, header

def print_rel_purity_results(pickle_results_file, nclusters_rng):
    rel_purities = results_rel_purity_per_dataset(pickle_results_file, nclusters_rng)
    tab_data = []

    views = list(rel_purities[list(rel_purities.keys())[0]]['20Newsgroup'])

    views.sort()

    for k in nclusters_rng:
        for ds in rel_purities[k]:
            view_row = [k, ds]
            for view in views:
                view_row.append(rel_purities[k][ds][view])
            tab_data.append(view_row.copy())
    header = ['k', 'dataset']
    header.extend(views)
    return tab_data, header


if __name__== "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--nruns', dest="NRUNS", type=int, default=10, required=True, help='an integer for the accumulator')
    parser.add_argument('--nclusters', dest="NCLUSTERS_RNG", type=int, nargs='+', help='an integer for the accumulator')
    parser.add_argument('--logfile', dest="LOGOUTPUTFILE", type=str, required=True, help='an integer for the accumulator')
    parser.add_argument('--ga', dest="GA",  action='store_true', required=False, help='Use genetic merging. Otherwise, NCI merging strategy is employed')
    parser.add_argument('--popsize', dest="POPSIZE", type=int, default=300, required=False, help='an integer for the accumulator')


    #####
    """
        Example usage: python NCF_clustering_fusion.py --nclusters 3 5 10 15 --nruns 5 --ga --logfile ga_ncf_run.log
    """
    ####


    args = parser.parse_args()
    #print(args)

    NCLUSTERS_RNG = args.NCLUSTERS_RNG
    NRUNS = args.NRUNS
    POPSIZE = args.POPSIZE
    #METHOD = "NCI" | "GA"
    METHOD = "NCI"
    if args.GA:
        METHOD = "GA"
    log_output_file = args.LOGOUTPUTFILE
    
    # EXECUTION PARAMETERS
    #NCLUSTERS_RNG = [15]
    #NRUNS = 2
    #POPSIZE = 300
    #METHOD = "GA":
    #METHOD = "NCI":
    #log_output_file = 'nci_ncf_run.log',


    logger = logging.getLogger('NCF clustering')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_output_file, mode='w')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    if args.GA:
        logger.info("Running experimentation with runs:{} #clusters in {} population size:{} and output log file {}".format(NRUNS, ' '.join(map(str, NCLUSTERS_RNG)), POPSIZE, log_output_file) )
    else:
        logger.info("Running experimentation with runs:{} #clusters in {} and output log file {}".format(NRUNS, ' '.join(map(str, NCLUSTERS_RNG)), log_output_file) )

    try:

        run_experimentation(NCLUSTERS_RNG, 
                            METHOD=METHOD, 
                            POPSIZE=POPSIZE, 
                            NRUNS=NRUNS,  
                            dataset_dir="../../data"
                            #dataset_dir="../Google Drive/Research - Multiview and Collaborative Clustering/data"
                            ) # dataset_dir parameter targets the dir where datasets are located.
        #outputfmt = 'latex'
        outputfmt = 'fancy_grid'
        
        logger.info("\nAverage Relative Entropy\n")
        tdata, header = print_ave_entropy_results('entropy_log.p',NCLUSTERS_RNG)
        logger.info("\n%s",tab(tdata, headers=header, tablefmt=outputfmt, floatfmt='.3f'))

        logger.info("\nRelative Entropy\n")
        tdata, header = print_rel_entropy_results('entropy_log.p',NCLUSTERS_RNG)
        logger.info("\n%s",tab(tdata, headers=header, tablefmt=outputfmt, floatfmt='.3f'))
        
        logger.info("\nAverage Relative Purity\n")
        tdata, header = print_ave_purity_results('entropy_log.p',NCLUSTERS_RNG)
        logger.info("\n%s",tab(tdata, headers=header, tablefmt=outputfmt, floatfmt='.3f'))

        logger.info("\nRelative Purity\n")
        tdata, header = print_rel_purity_results('entropy_log.p',NCLUSTERS_RNG)
        logger.info("\n%s",tab(tdata, headers=header, tablefmt=outputfmt, floatfmt='.3f'))
    except KeyboardInterrupt:
        logger.info("Abruptly finished!")
        sys.exit()
        pass
