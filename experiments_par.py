# 26-07-2020
# TODO: End the evaluation procedure!
# TODO: (Main) Assess the stability of the view ranks in NCFwR
# TODO: Answer how the random partitions affect the 'Bias'
# TODO: Answer how many random partitions are neccessary for the Method to obtain the same rank or performance result
# For each dataset:
# # for each method M
# # # for each k, repeat the procedure and pick the best entropy/purity
# # Pick the best value among all the methods
# # # relative-entropy ~ Divide each ave-entropy by smallest entropy
# # # relative-purity ~ Divide the best purity by each ave-purity

from original_ncf import NCFwR, NCF, BadSourcePartitionException
import utils
from logging_setup import logger
from tabulate import tabulate as tab
from data_source import TwentyNewsgroupView, BBCSportsView, ReutersView, WEBKBView
import numpy as np
import json
from datetime import datetime
from time import perf_counter
from math import ceil
from scoop import futures, shared
from itertools import product
import importlib

def scoop_perform_single_run(params):
    """
    executes a single run by parsing and evaluating the expressions in params.
    :param params: list with factors and parameters. E.g. (intK, strMethodClassWithArgs, strDatasetClass)
    :return: a dict with the results obtained for all the runs and for each view of the DSet.
    """
    datapath = shared.getConst("datapath")
    initial_seed = shared.getConst("initial_seed")
    np.random.seed(initial_seed)
    nruns = shared.getConst("nruns")

    logger.debug("parallel run {0}".format(params))

    # parsing the parameters
    k_val = params[0] # number of clusters to set for the views
    met_op_str = params[1].split(":")[0]
    met_op_args_str = params[1].split(":")[1:]
    argsDict = {'seed' : None}
    for arg_i in met_op_args_str:
        #logger.debug("adding arg %s" % (arg_i.split("=")))
        arg_name, arg_val = arg_i.split("=")
        logger.debug("adding arg key {0} -> {0}".format(arg_name,arg_val))
        argsDict[arg_name] = int(arg_val) # this must be specified in a more flexible way
    #logger.debug("Params parsed: %s" % (argsDict))
    # getting the required classes
    ds_mod = importlib.import_module("data_source")
    ds_class_ = getattr(ds_mod, params[2])
    met_op_mod = importlib.import_module("original_ncf")
    met_op_class_ = getattr(met_op_mod, met_op_str)
    # starting the loop of runs
    results_per_run = {}
    logger.debug("run with params:{0}".format(params))
    for r in range(nruns):
        seed = np.random.randint(1, 1e5)
        ds_inst = ds_class_(datapath, k_val, seed) #  instances the datasource

        argsDict['seed'] = seed
        logger.debug("Creating method {0}".format(met_op_str))
        met_op_inst = met_op_class_(args=argsDict) # instances the method
        met_name = met_op_inst.getName()
        met_op_inst.setInputPartitions(ds_inst.get_views())
        logger.debug("Created!")
        if not met_name in results_per_run:
            results_per_run[met_name] = {}

        try:
            consensus_partition = met_op_inst.run() # executes the method!
            consensus_kval = len(np.unique(consensus_partition))

            consensus_E = utils.Entropy(consensus_partition, ds_inst.get_real_labels())
            consensus_P = utils.Purity(consensus_partition, ds_inst.get_real_labels())

            dsname_kval = "{0}:{1}".format(ds_inst.name, k_val)  # nr of clusters is appended to the dataset name for visualization purposes.
            if not dsname_kval in results_per_run[met_name]:
                results_per_run[met_name][dsname_kval] = {}
                for viewname in ds_inst.data_views:
                    results_per_run[met_name][dsname_kval][viewname] = {"K": ds_inst.NCLUSTERS[viewname], "entropy": [],
                                                                "purity": []}
                results_per_run[met_name][dsname_kval]["consensus"] = {"K": consensus_kval, "entropy": [], "purity": []}

            for viewname in ds_inst.data_views:
                v_E = utils.Entropy(ds_inst.get_views()[viewname], ds_inst.get_real_labels())
                v_P = utils.Purity(ds_inst.get_views()[viewname], ds_inst.get_real_labels())
                results_per_run[met_name][dsname_kval][viewname]["entropy"].append(v_E)
                results_per_run[met_name][dsname_kval][viewname]["purity"].append(v_P)

            results_per_run[met_name][dsname_kval]["consensus"]["entropy"].append(consensus_E)
            results_per_run[met_name][dsname_kval]["consensus"]["purity"].append(consensus_P)

            if hasattr(met_op_inst, 'complexity_rank'):
                if not 'aveK_ranks' in results_per_run[met_name][dsname_kval]:
                    results_per_run[met_name][dsname_kval]['aveK_ranks'] = []
                results_per_run[met_name][dsname_kval]['aveK_ranks'].append(list(met_op_inst.complexity_rank))
            #logger.debug("run: %s" % (params))
            #logger.debug("Performance of %s" % (dsname_kval))
            #logger.debug("Entropy: %f" % (consensus_E))
            #logger.debug("Purity: %f" % (consensus_P))

        except BadSourcePartitionException as e:
            logger.error(e)

    return results_per_run



if __name__ == '__main__':
    #data = "C:/Users/juan/Insync/juan.zamora@pucv.cl/Google Drive/Research - Multiview and Collaborative Clustering/data"
    # data = "D:/Google Drive/Research - Multiview and Collaborative Clustering/data"
    #shared.setConst(datapath="C:/Users/juan/Insync/juan.zamora@pucv.cl/Google Drive/Research - Multiview and Collaborative Clustering/data")
    shared.setConst(datapath="data")
    shared.setConst(initial_seed=1000982)
    shared.setConst(nruns=10)

    k_values = [3, 6, 12, 24, 48]
    #k_values = [3, 6]
    #methods = ["NCF", "NCFwR:number_random_partitions=10", "NCFwR:number_random_partitions=20"]
    #datasources = ["TwentyNewsgroupView", "BBCSportsView"]
    methods = ["NCF", "NCFwR:number_random_partitions=10", "NCFwR:number_random_partitions=20",
               "NCFwR:number_random_partitions=30", "NCFwR:number_random_partitions=60",
               "NCFwR:number_random_partitions=80", "NCFwR:number_random_partitions=100"]
    datasources = ["TwentyNewsgroupView", "BBCSportsView", "ReutersView", "WEBKBView"]
    computation_lst = list(product(*[k_values, methods, datasources]))

    start_time = perf_counter()
    logger.debug("Starting parallel procedure...")

    result_lst = list(futures.map(scoop_perform_single_run, computation_lst))

    # storage routines
    end_time = perf_counter()
    elapsed = ceil(end_time - start_time)
    logger.info("Overall procedure ended.")
    now = datetime.now()
    outputfile = "RES_{0}_{1}secs.json".format(now.strftime("%b%d%Y.%H%M%S"), elapsed)

    results = {}
    for run_ex in result_lst:
        assert len(run_ex) == 1
        logger.debug(("KEYS in RUN {0}".format(run_ex.keys())))
        method = list(run_ex.keys())[0]
        ds_dict = run_ex[method]
        assert len(ds_dict) == 1
        datasrc = list(ds_dict.keys())[0]
        if not method in results:
            results[method] = {}
        if 'aveK_ranks' in ds_dict[datasrc]:
            ds_dict[datasrc]['aveK_ranks'] = [','.join(map(str,ranks)) for ranks in ds_dict[datasrc]['aveK_ranks']]
        results[method][datasrc] = ds_dict[datasrc]

    logger.info("Elapsed time {0:.3f} secs".format(end_time - start_time))

    with open(outputfile, 'w') as fp:
        json.dump(results, fp)
    logger.info("Results stored into file {0}".format(outputfile))
