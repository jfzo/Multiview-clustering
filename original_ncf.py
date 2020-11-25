import logging
from multiprocessing import Value
import numpy as np
from logging_setup import logger
from utils import *


class NCFwR(object):
    def __init__(self, *args, **kwargs):
        """
        Necessary for using in an automatic execution procedure
        :param dict_args: dict with argument names as keys and values.
        """
        self.N_rnd=50
        self.seed = 1
        if "args" in kwargs:
            dict_args = kwargs["args"]
            logger.debug("NCFwR params:{0}".format(dict_args))
            if "number_random_partitions" in dict_args:
                self.N_rnd = dict_args["number_random_partitions"]
            if "seed" in dict_args:
                self.seed = dict_args["seed"]

        self.logger = logger
        logger.debug("Setting the name")
        self.__method_name = "NCFwR#{0}".format( self.N_rnd)
        logger.debug("Starting {0}...".format(self.__method_name))

        # capital Pi: list of partitionings.
        # A partitioning is a set of integers (data point id)
        # logger.debug(str(input_partitions.keys()))
        self.Pi = None
        self.complexity_rank = None


    def getName(self):
        return self.__method_name

    def setInputPartitions(self,  input_partitions):
        self.Pi = [x for name,x in input_partitions.items()]


    def run(self):
        """
        result: Partitionings as integer vectors. One vector for each view.
        """
        #result = self.partitions
        if self.Pi is None:
            logger.error("input partitions must be set first")
            raise ValueError

        logger.debug("Executing NCI proposal")
        REFERENCE_VALUE = float('Inf') # this value is used for minimum-distance merging
            
        
        m = len(self.Pi) # nr. of views
        
        logger.debug("initial number of views:".format(m) )
        K = [np.unique(k).shape[0] for k in self.Pi]

        n = len(self.Pi[0])

        np.random.seed(seed=self.seed)
        kmax = np.max(K)
        rnd_p = [random_partition(kmax, n) for i in range(self.N_rnd)]
        # ranking
        # for each view, the average complexity against all random clusterings is computed.
        ave_complexity = [np.mean([compute_solutions_complexity(self.Pi[j], rnd_p[i], K[j], kmax)[2] for i in range(self.N_rnd)]) for j in range(m)]
        # if ave_complexity[i] > ave_complexity[j] => view_i has more information thant view_j, thus it is ranked higher
        #logger.debug("Average Complexities under #rndcst={0}".format(self.N_rnd))
        #logger.debug("{0}".format(ave_complexity) )
        self.complexity_rank = np.argsort(ave_complexity) # sorts ave complexities in ascending order (recall that the higher the farther from random)
        ######
        # Distance matrix computation
        #
        Aff = np.repeat(REFERENCE_VALUE, m**2).reshape(m, m)

        for i in range(m-1):
            for j in range(i+1,m):
                _,_,val1 = compute_solutions_complexity(self.Pi[i], self.Pi[j], K[i], K[j])
                _,_,val2 = compute_solutions_complexity(self.Pi[j], self.Pi[i], K[j], K[i])
                #if(len(set(e1.keys()) & set(e2.keys())) != len(set(e1.keys()) | set(e2.keys())) ):
                #    print("C1:{0} C2:{0}".format(i,j))
                #else:
                #    print("*C1:{0} C2:{0}".format(i,j))
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

            #print("merge views {0} and {0}".format(mindist_partitions, row_min_dist) )
            #merging_dist = np.min(np.min(Aff, axis=1))
            #newclustering, exceptions, memberships_ = merge(mindist_partitions, row_min_dist, self.Pi, K, step_memberships, alpha=0.1)

            optimal_col_per_row = np.argmin(Aff, axis=1) # for each row, which column presents the lowest distance (BY ROW)
            optimal_val_per_row = np.min(Aff, axis=1)
            optimal_row = np.argmin(optimal_val_per_row) # row whose min distance is the global min.
            optimal_col = optimal_col_per_row[optimal_row] # combining the previous commands, obtain the least distant views to merge.

            #merging_dist = np.min(np.min(Aff, axis=1))

            #print(Aff)

            # original version had only the following line
            #newclustering, exceptions = merge(optimal_row, optimal_col, self.Pi, K, exception_weights, alpha=0.1)
            # now the ranked version
            if ave_complexity[optimal_row] > ave_complexity[optimal_col]:
                logger.debug("Merging views {0} with {1}".format(optimal_row, optimal_col))
                newclustering, exceptions = self.merge(optimal_row, optimal_col, K, exception_weights, alpha=0.1)            
            else:
                logger.debug("Merging views {0} with {1}".format(optimal_col, optimal_row))
                newclustering, exceptions = self.merge(optimal_col, optimal_row, K, exception_weights, alpha=0.1)
            
            #step_memberships.append(memberships_)
            K.append(np.unique(newclustering[np.where(newclustering >= 0)[0]]).shape[0]) 
            self.Pi.append(newclustering.copy())
            newclustering_index = len(self.Pi)-1
            last_created_view = newclustering_index
            exception_weights[newclustering_index] = exceptions
            
            # rank for the new view
            new_complexity = np.mean([compute_solutions_complexity(self.Pi[newclustering_index], rnd_p[i], K[newclustering_index], kmax)[2] for i in range(self.N_rnd)])
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

            #print("1st partition:",np.unique(self.Pi[newclustering_index]))
            # Update the Kolmogorov complexities
            for j in range(len(self.Pi)):
                if j in processed_views or j == newclustering_index:
                    continue
                _,_,val1 = compute_solutions_complexity(self.Pi[newclustering_index], self.Pi[j], K[newclustering_index], K[j])
                _,_,val2 = compute_solutions_complexity(self.Pi[j], self.Pi[newclustering_index], K[j], K[newclustering_index])

                Aff[newclustering_index,j] = val1 + val2
                Aff[j,newclustering_index] = Aff[newclustering_index,j]

            #update the m value
            m = len(self.Pi)

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

        nclusters = np.unique(self.Pi[last_created_view]).shape[0]
        D = np.zeros((n,nclusters))
        for i in range(len(self.Pi[last_created_view])):
            if self.Pi[last_created_view][i] != -1:
                D[i, self.Pi[last_created_view][i]] = 1

        #alreadyAssignedPts = np.where(self.Pi[last_created_view] != -1)[0]
        #D[alreadyAssignedPts, self.Pi[last_created_view][alreadyAssignedPts] ] = 1

        for ((p_id,c_id), w) in exception_weights[last_created_view].items():
            assert self.Pi[last_created_view][p_id] == -1 # len(self.Pi)-1 in line 285

            D[p_id, c_id] = w
            if not p_id in max_weight_cluster:
                max_weight_cluster[p_id] = (c_id, w)
            elif max_weight_cluster[p_id][1] < w:
                max_weight_cluster[p_id] = (c_id, w)
            
        for (p_id, (c,_)) in  max_weight_cluster.items():
            self.Pi[last_created_view][p_id] = c
        
        return self.Pi[last_created_view], D


    def merge(self, i, j, K, exception_weights, alpha=0.7):
        """
        i,j : pair of views to merge
        self.Pi : Collection with the partitionings (1D vector for each view)
        K : Collection with the number of clusters per view.  
        
        Returns:
        marked_point_exceptions: Dict with tuples (point_id-cluster_id) as keys and weights as values.
        alpha: weight of the historic membership accumulated. The weight of the current membership is (1-alpha)
        Note
        ====
        Label -1 identifies points marked as exceptions (is not a cluster) so we take it out.
        """
        labels = list((set(np.unique(self.Pi[i])) |set(np.unique(self.Pi[j]))  ) - {-1})
        
        r1,_,_ = compute_solutions_complexity(self.Pi[i], self.Pi[j], K[i], K[j], labels=labels)
        r2,_,_ = compute_solutions_complexity(self.Pi[j], self.Pi[i], K[j], K[i], labels=labels)
        #print("r1:",r1)
        #print("r2:",r2)
        #[(x,self.Pi[i][x],self.Pi[j][x]) for x in (set(e1.keys()) | set(e2.keys()))-(set(e1.keys()) & set(e2.keys()))]

        merge_ops = dict([(u,set()) for u in range(len(r1))])
        for u in range(len(r1)):
            #print("MERGE({0}, {0})".format(u, r1[u]))
            merge_ops[u].add(r1[u])

        for v in range(len(r2)):
            #print("MERGE({0}, {0})".format(v, r2[v]))
            merge_ops[r2[v]].add(v)
        logger.debug(merge_ops)
        
        cache = {} #employed for storing the updated versions of the partitions not merged.
        #new_partitioning = np.zeros(self.Pi[i].shape, dtype=np.int32)
        new_partitioning = np.repeat(-1, self.Pi[i].shape[0]) # By default each pt is marked.
        cluster_id = 0 # Cluster ids start from 0

        for e,s in merge_ops.items():
            # candidate points to merge initially
            set_vie = set(np.where(self.Pi[i] == e)[0])
            # CREATE A NEW CLUSTER
            c = set()
            for p in s: # p is an integer denoting a cluster in view j 
                # add to c only candidates & p
                set_vjp = None
                if p in cache:
                    set_vjp = cache[p] # not a copy
                else:
                    set_vjp = set(np.where(self.Pi[j] == p)[0])
                c |= (set_vie & set_vjp) # adding intersection to the new cluster
                a = set_vie - set_vjp # the remaining items to merge for next p's in s
                # Update set_vjp to (set_vjp - set_vie ) to use in further (e,s)            
                cache[p] = set_vjp - set_vie
                set_vie = set(a)

            # when there are no common data points, the merge generates only exceptions.
            if len(c) == 0:
                logger.error("ERROR empty cluster when merging views i={} j={}: No common points between partition {} and {}".format(i,j,e,s))
                logger.warning("|view {} partition {}| : {}".format(i,e,len(set_vie)))
                for pt in s:
                    logger.warning("|view {} partition {}| : {}({})".format(j,pt,len(cache[pt]),len(set(np.where(self.Pi[j] == pt)[0])) ) )
                # maybe this issue should throw an exception since the result is no longer valid (Bad source clustering).
                #assert len(c) > 0
                raise BadSourcePartitionException(e,s)

                
            new_partitioning[np.array(list(c), dtype=np.int32)] = cluster_id
            cluster_id += 1
            # end of the creation of the new partition (for e,s ...)
            
        # this var contains all the points marked as exceptions in the current merge and in the merges made when
        # the source views were created.
        overall_exceptions = set(np.where(new_partitioning == -1)[0])

        # computing the weights of marked points
        marked_points_weights = {}
        #for p in overall_candidates: # put also the past exceptions!
        for p in overall_exceptions:  # put also the past exceptions!
            available_labels = np.unique(new_partitioning[np.where(new_partitioning > -1)[0]])
            epweights_A = exception_weights.get(i, {}) # original source views dont have exceptions
            epweights_B = exception_weights.get(j, {})
            w_a = [omega(set(np.where(new_partitioning == l)[0]), p, self.Pi[i], epweights_A) for l in available_labels]
            w_b = [omega(set(np.where(new_partitioning == l)[0]), p, self.Pi[j], epweights_B) for l in available_labels]
            
            for l_ix in range(len(available_labels)): # generally, the l_ix must match the label id.
                W_pl = 0.5*w_a[l_ix] + 0.5*w_b[l_ix]
                marked_points_weights[(p, available_labels[l_ix])] = W_pl
            
        return new_partitioning, marked_points_weights


class BadSourcePartitionException(Exception):
    def __init__(self, e: int, s: set):
        self.s = s
        self.e = e

    def __str__(self):
        return "Empty merging for view #{0} ~ {0}".format(self.e, self.s)



class NCF(object):
    def __init__(self, *args, **kwargs):
        """
        input_partitions: Dictionary with
        partition-name:list of cluster assignments.
        """
        # self.logger = logging.getLogger(__name__)
        self.logger = logger
        self.__method_name = "NCF"
        logger.debug("Starting {0}...".format(self.__method_name))
        # capital Pi: list of partitionings.
        # A partitioning is a set of integers (data point id)
        # logger.debug(str(input_partitions.keys()))

    def getName(self):
        return self.__method_name

    def setInputPartitions(self, input_partitions):
        self.Pi = [x for name, x in input_partitions.items()]

    def run(self) -> object:
        """
        result: Partitionings as integer vectors. One vector for each view.
        """
        # result = self.partitions

        logger.debug("Executing NCI proposal")
        if self.Pi is None:
            logger.error("input partitions must be set first")
            raise ValueError

        REFERENCE_VALUE = float('Inf')  # this value is used for minimum-distance merging

        m = len(self.Pi)  # nr. of views

        logger.debug("initial number of views:".format(m))
        K = [np.unique(k).shape[0] for k in self.Pi]


        ######
        # Distance matrix computation
        #
        Aff = np.repeat(REFERENCE_VALUE, m ** 2).reshape(m, m)

        for i in range(m - 1):
            for j in range(i + 1, m):
                _, _, val1 = compute_solutions_complexity(self.Pi[i], self.Pi[j], K[i], K[j])
                _, _, val2 = compute_solutions_complexity(self.Pi[j], self.Pi[i], K[j], K[i])
                # if(len(set(e1.keys()) & set(e2.keys())) != len(set(e1.keys()) | set(e2.keys())) ):
                #    print("C1:{0} C2:{0}".format(i,j))
                # else:
                #    print("*C1:{0} C2:{0}".format(i,j))
                Aff[i, j] = val1 + val2
                Aff[j, i] = Aff[i, j]
        # print(Aff)

        processed_views = set()  # stores the partitionings already merged so they are not re-used
        # marked_pts = {} # dictionary with point ids (integer) as keys and list of tuples (cluster-id; weight)
        #
        # the following must be repeated until no more partitionings can be merged.
        # It seems that the condition is that the min value of the distance matrix is Inf.
        exception_weights = {}  # dict with View_id as key and a dict ((point_id,cluster_id)->weight) as value.
        n_its = 0
        last_created_view = -1
        while (not np.isinf(np.min(Aff))):
            # mindist_part_for_rows = np.argmin(Aff, axis=1) # for each row, which column presents the lowest distance
            # mindist_partitions = np.argmin(np.min(Aff, axis=1)) # row whose min distance is the global min.
            # row_min_dist = mindist_part_for_rows[mindist_partitions] # combining the previous commands, obtain the least distant views to merge.

            # print("merge views {0} and {0}".format(mindist_partitions, row_min_dist) )
            # merging_dist = np.min(np.min(Aff, axis=1))
            # newclustering, exceptions, memberships_ = merge(mindist_partitions, row_min_dist, self.Pi, K, step_memberships, alpha=0.1)

            optimal_col_per_row = np.argmin(Aff, axis=1)  # for each row, which column presents the lowest distance
            optimal_val_per_row = np.min(Aff, axis=1)
            optimal_row = np.argmin(optimal_val_per_row)  # row whose min distance is the global min.
            optimal_col = optimal_col_per_row[
                optimal_row]  # combining the previous commands, obtain the least distant views to merge.

            # merging_dist = np.min(np.min(Aff, axis=1))

            # print(Aff)


            #newclustering, exceptions = merge(optimal_row, optimal_col, Pi, K, exception_weights, alpha=0.1)
            logger.debug("Merging views {0} with {0}".format(optimal_row, optimal_col))
            newclustering, exceptions = self.merge(optimal_row, optimal_col, K, exception_weights, alpha=0.1)

            # step_memberships.append(memberships_)
            K.append(np.unique(newclustering[np.where(newclustering >= 0)[0]]).shape[0])
            self.Pi.append(newclustering.copy())
            newclustering_index = len(self.Pi) - 1
            last_created_view = newclustering_index
            exception_weights[newclustering_index] = exceptions

            logger.debug(
                "merge views {} and {} --> {} is created.".format(optimal_row, optimal_col, newclustering_index))

            # TODO: add exceptions to the overall list
            # for e,_ in exceptions:
            #    if e not in marked_pts:
            #        marked_pts[e] = []
            #    marked_pts[e].append((newclustering_index, 1/merging_dist))

            # set the rows/columns of merged partitions as processed!
            processed_views.add(optimal_row)
            processed_views.add(optimal_col)

            # Update the distance matrix: Set the merged partitionings distances to infinity so they
            # are not employed in posterior iterations ~ Removing these partitions from the set of candidates
            Aff[optimal_col, :] = np.repeat(REFERENCE_VALUE, m)
            Aff[:, optimal_col] = np.repeat(REFERENCE_VALUE, m)
            Aff[optimal_row, :] = np.repeat(REFERENCE_VALUE, m)
            Aff[:, optimal_row] = np.repeat(REFERENCE_VALUE, m)

            # Include the new partitioning into the set of views:
            #    * A row and column are added to represent the new clustering.
            #    * Distances between this new partitioning and the valid ones are computed.
            Aff = np.vstack((
                np.hstack((
                    Aff, np.repeat(REFERENCE_VALUE, Aff.shape[0]).reshape(Aff.shape[0], 1))),
                np.repeat(REFERENCE_VALUE, Aff.shape[0] + 1)))

            # print("1st partition:",np.unique(self.Pi[newclustering_index]))
            # Update the Kolmogorov complexities
            for j in range(len(self.Pi)):
                if j in processed_views or j == newclustering_index:
                    continue
                _, _, val1 = compute_solutions_complexity(self.Pi[newclustering_index], self.Pi[j],
                                                          K[newclustering_index], K[j])
                _, _, val2 = compute_solutions_complexity(self.Pi[j], self.Pi[newclustering_index], K[j],
                                                          K[newclustering_index])

                Aff[newclustering_index, j] = val1 + val2
                Aff[j, newclustering_index] = Aff[newclustering_index, j]

            # update the m value
            m = len(self.Pi)

            n_its += 1
            if n_its > 50:
                logger.error("Nr. of max iterations reached. Halting!")
                break

        # solving the exceptions

        logger.debug("Solving the exceptions!")

        max_weight_cluster = {}
        # Since the exceptions are carried, only consider the last view weights
        # print("Solving the exceptions...\n ##################################")
        # print(exceptions[last_created_view],"\n ###############################")
        for ((p_id, c_id), w) in exception_weights[last_created_view].items():
            assert self.Pi[last_created_view][p_id] == -1  # len(self.Pi)-1 in line 285

            if not p_id in max_weight_cluster:
                max_weight_cluster[p_id] = (c_id, w)
            elif max_weight_cluster[p_id][1] < w:
                max_weight_cluster[p_id] = (c_id, w)

        for (p_id, (c, _)) in max_weight_cluster.items():
            self.Pi[last_created_view][p_id] = c

        return self.Pi[last_created_view], None

    def merge(self, i, j, K, exception_weights, alpha=0.7):
        """
        i,j : pair of views to merge
        self.Pi : Collection with the partitionings (1D vector for each view)
        K : Collection with the number of clusters per view.

        Returns:
        marked_point_exceptions: Dict with tuples (point_id-cluster_id) as keys and weights as values.
        alpha: weight of the historic membership accumulated. The weight of the current membership is (1-alpha)
        Note
        ====
        Label -1 identifies points marked as exceptions (is not a cluster) so we take it out.
        """
        labels = list((set(np.unique(self.Pi[i])) | set(np.unique(self.Pi[j]))) - {-1})

        r1, _, _ = compute_solutions_complexity(self.Pi[i], self.Pi[j], K[i], K[j], labels=labels)
        r2, _, _ = compute_solutions_complexity(self.Pi[j], self.Pi[i], K[j], K[i], labels=labels)
        # print("r1:",r1)
        # print("r2:",r2)
        # [(x,self.Pi[i][x],self.Pi[j][x]) for x in (set(e1.keys()) | set(e2.keys()))-(set(e1.keys()) & set(e2.keys()))]

        merge_ops = dict([(u, set()) for u in range(len(r1))])
        for u in range(len(r1)):
            # print("MERGE({0}, {0})".format(u, r1[u]))
            merge_ops[u].add(r1[u])

        for v in range(len(r2)):
            # print("MERGE({0}, {0})".format(v, r2[v]))
            merge_ops[r2[v]].add(v)
        logger.debug(merge_ops)

        cache = {}  # employed for storing the updated versions of the partitions not merged.
        # new_partitioning = np.zeros(self.Pi[i].shape, dtype=np.int32)
        new_partitioning = np.repeat(-1, self.Pi[i].shape[0])  # By default each pt is marked.
        cluster_id = 0  # Cluster ids start from 0

        for e, s in merge_ops.items():
            # candidate points to merge initially
            set_vie = set(np.where(self.Pi[i] == e)[0])
            # CREATE A NEW CLUSTER
            c = set()
            for p in s:  # p is an integer denoting a cluster in view j
                # add to c only candidates & p
                set_vjp = None
                if p in cache:
                    set_vjp = cache[p]  # not a copy
                else:
                    set_vjp = set(np.where(self.Pi[j] == p)[0])
                c |= (set_vie & set_vjp)  # adding intersection to the new cluster
                a = set_vie - set_vjp  # the remaining items to merge for next p's in s
                # Update set_vjp to (set_vjp - set_vie ) to use in further (e,s)
                cache[p] = set_vjp - set_vie
                set_vie = set(a)

            # when there are no common data points, the merge generates only exceptions.
            if len(c) == 0:
                logger.error("ERROR empty cluster when merging views i={} j={}: No common \
    points between partition {} and {}".format(i, j, e, s))
                logger.warning("|view {} partition {}| : {}".format(i, e, len(set_vie)))
                for pt in s:
                    logger.warning("|view {} partition {}| : {}({})".format(j, pt, len(cache[pt]),
                                                                            len(set(np.where(self.Pi[j] == pt)[0]))))
                # maybe this issue should throw an exception since the result is no longer valid (Bad source clustering).
                #assert len(c) > 0
                raise BadSourcePartitionException(e,s)

            new_partitioning[np.array(list(c), dtype=np.int32)] = cluster_id
            cluster_id += 1
            # the remaining points in View i that couldn't be merged are marked as potential exceptions.
            #overall_candidates |= set_vie
            # end of the creation of the new partition

        # Additionally, Points that couldn't be included in any merge must also be marked.
        overall_exceptions = set(np.where(new_partitioning == -1)[0])

        # computing the weights of marked points
        marked_points_weights = {}
        # for p in overall_candidates: # put also the past exceptions!
        for p in overall_exceptions:
            available_labels = np.unique(new_partitioning[np.where(new_partitioning > -1)[0]])
            epweights_A = exception_weights.get(i, {})  # original source views dont have exceptions
            epweights_B = exception_weights.get(j, {})
            w_a = [omega(set(np.where(new_partitioning == l)[0]), p, self.Pi[i], epweights_A) for l in available_labels]
            w_b = [omega(set(np.where(new_partitioning == l)[0]), p, self.Pi[j], epweights_B) for l in available_labels]

            for l_ix in range(len(available_labels)):  # generally, the l_ix must match the label id.
                W_pl = 0.5 * w_a[l_ix] + 0.5 * w_b[l_ix]
                marked_points_weights[(p, available_labels[l_ix])] = W_pl

        return new_partitioning, marked_points_weights
