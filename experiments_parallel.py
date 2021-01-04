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
import numpy as np
import json
import pickle
from datetime import datetime
from time import perf_counter
from math import ceil
from itertools import product
import importlib
from multiprocessing import Pool, set_start_method
import os
from sklearn import metrics


def perform_single_run(params):
    """
    executes a single run by parsing and evaluating the expressions in params.
    :param params: list with factors and parameters. E.g. (intK, strMethodClassWithArgs, strDatasetClass)
    :return: a dict with the results obtained for all the runs and for each view of the DSet.
    """
    np.random.seed(initial_seed)
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
    #ds_mod = importlib.import_module("data_source")
    #ds_mod = importlib.import_module("multiview_datasets")

    ds_mod = importlib.import_module("HDF5Datasets")
    ds_class_ = getattr(ds_mod, params[2])

    met_op_mod = importlib.import_module("original_ncf")
    met_op_class_ = getattr(met_op_mod, met_op_str)

    # starting the loop of runs
    results_per_run = {}
    logger.debug("run with params:{0}".format(params))
    for r in range(nruns):
        seed = np.random.randint(1, 1e5)
        #ds_inst = ds_class_(datapath, k_val, seed) #  instances the datasource
        ds_inst = ds_class_(path=datapath)
        dsname_kval = "{0}:{1}".format(ds_inst.name,
                                       k_val)  # nr of clusters is appended to the dataset name for visualization purposes.
        argsDict['seed'] = seed
        logger.debug("Creating method {0}".format(met_op_str))

        met_op_inst = met_op_class_(args=argsDict) # Instances the dataset handler
        met_name = met_op_inst.getName()
        met_op_inst.setInputPartitions(ds_inst.get_views())

        logger.debug("Created!")
        if not met_name in results_per_run:
            results_per_run[met_name] = {}
            results_per_run[met_name][dsname_kval] = {}
            for viewname in ds_inst.get_views():
                ncusters_in_view = np.unique(ds_inst.get_views()[viewname])
                results_per_run[met_name][dsname_kval][viewname] = {"K": ncusters_in_view,
                                                                    "entropy": [],
                                                                    "purity": [],
                                                                    "f1" : [],
                                                                    "accuracy": [],
                                                                    "nmi" : [],
                                                                    "precision" : [],
                                                                    "recall" : [],
                                                                    "ari" : []}
            results_per_run[met_name][dsname_kval]["consensus"] = {"K": None,
                                                                   "entropy": [],
                                                                   "purity": [],
                                                                   "f1": [],
                                                                   "accuracy": [],
                                                                   "nmi": [],
                                                                   "precision": [],
                                                                   "recall": [],
                                                                   "ari": []}
        try:
            consensus_partition, D = met_op_inst.run() # executes the method!
            # build postfix with the date
            # now = datetime.now()
            # outputPostfix = "{0}".format(now.strftime("%b%d%Y.%H%M%S"))
            # np.savetxt('labels_NCF_DS_{0}.out'.format(outputPostfix), consensus_partition, delimiter=',', fmt='%d')
            # np.savetxt('simmatrix_DS_{0}.out'.format(outputPostfix), D, delimiter=',', fmt='%.4f')

            consensus_kval = len(np.unique(consensus_partition))
            results_per_run[met_name][dsname_kval]["consensus"]["K"] = consensus_kval

            consensus_E = utils.Entropy(consensus_partition, ds_inst.get_real_labels())
            consensus_P = utils.Purity(consensus_partition, ds_inst.get_real_labels())
            consensus_F1 = utils.F1Score(consensus_partition, ds_inst.get_real_labels())
            consensus_ACC = utils.ACCScore(consensus_partition, ds_inst.get_real_labels())
            consensus_NMI = utils.NMIScore(consensus_partition, ds_inst.get_real_labels())
            consensus_PREC = utils.PRECScore(consensus_partition, ds_inst.get_real_labels())
            consensus_REC = utils.RECScore(consensus_partition, ds_inst.get_real_labels())
            consensus_ARI = utils.ARIScore(consensus_partition, ds_inst.get_real_labels())

            for viewname in ds_inst.get_views():
                v_E = utils.Entropy(ds_inst.get_views()[viewname], ds_inst.get_real_labels())
                results_per_run[met_name][dsname_kval][viewname]["entropy"].append(v_E)
                v_P = utils.Purity(ds_inst.get_views()[viewname], ds_inst.get_real_labels())
                results_per_run[met_name][dsname_kval][viewname]["purity"].append(v_P)
                v_F1 = utils.F1Score(ds_inst.get_views()[viewname], ds_inst.get_real_labels())
                results_per_run[met_name][dsname_kval][viewname]["f1"].append(v_F1)
                v_ACC = utils.ACCScore(ds_inst.get_views()[viewname], ds_inst.get_real_labels())
                results_per_run[met_name][dsname_kval][viewname]["accuracy"].append(v_ACC)
                v_NMI = utils.NMIScore(ds_inst.get_views()[viewname], ds_inst.get_real_labels())
                results_per_run[met_name][dsname_kval][viewname]["nmi"].append(v_NMI)
                v_PREC = utils.PRECScore(ds_inst.get_views()[viewname], ds_inst.get_real_labels())
                results_per_run[met_name][dsname_kval][viewname]["precision"].append(v_PREC)
                v_REC = utils.RECScore(ds_inst.get_views()[viewname], ds_inst.get_real_labels())
                results_per_run[met_name][dsname_kval][viewname]["recall"].append(v_REC)
                v_ARI = utils.ARIScore(ds_inst.get_views()[viewname], ds_inst.get_real_labels())
                results_per_run[met_name][dsname_kval][viewname]["ari"].append(v_ARI)

            results_per_run[met_name][dsname_kval]["consensus"]["entropy"].append(consensus_E)
            results_per_run[met_name][dsname_kval]["consensus"]["purity"].append(consensus_P)
            results_per_run[met_name][dsname_kval]["consensus"]["f1"].append(consensus_F1)
            results_per_run[met_name][dsname_kval]["consensus"]["accuracy"].append(consensus_ACC)
            results_per_run[met_name][dsname_kval]["consensus"]["nmi"].append(consensus_NMI)
            results_per_run[met_name][dsname_kval]["consensus"]["precision"].append(consensus_PREC)
            results_per_run[met_name][dsname_kval]["consensus"]["recall"].append(consensus_REC)
            results_per_run[met_name][dsname_kval]["consensus"]["ari"].append(consensus_ARI)

            if hasattr(met_op_inst, 'complexity_rank'):
                if not 'aveK_ranks' in results_per_run[met_name][dsname_kval]:
                    results_per_run[met_name][dsname_kval]['aveK_ranks'] = []
                results_per_run[met_name][dsname_kval]['aveK_ranks'].append(list(met_op_inst.complexity_rank))
            #logger.debug("run: %s" % (params))
            #logger.debug("Performance of %s" % (dsname_kval))
            #logger.debug("Entropy: %f" % (consensus_E))
            #logger.debug("Purity: %f" % (consensus_P))

        except BadSourcePartitionException as e:
            dsname_kval = "{0}:{1}".format(ds_inst.name, k_val)  # nr of clusters is appended to the dataset name for visualization purposes.
            if not dsname_kval in results_per_run[met_name]:
                results_per_run[met_name][dsname_kval] = {}
            logger.error(e)

    return results_per_run


def initializer(arg0, arg1, arg2):
    """
    Initializer function called by each Worker process.
    The only thing this method does is to make three values as global variables. Namely:
    :param arg0: datapath (folder where the input data is stored)
    :param arg1: initial seed value for reproducibility.
    :param arg2: number of runs per experiment.
    :return: None.
    """
    global datapath
    datapath = arg0
    global initial_seed
    initial_seed = arg1
    global nruns
    nruns = arg2



if __name__ == '__main__':
    # data = "C:/Users/juan/Insync/juan.zamora@pucv.cl/Google Drive/Research - Multiview and Collaborative Clustering/data"
    # data = "D:/Google Drive/Research - Multiview and Collaborative Clustering/data"

    set_start_method("spawn")

    datapath="C:/Users/juan/Documents/multiview-data"
    #datapath = "/home/juan/Documentos/sparse-multiview-data"
    initial_seed=111
    nruns=10

    k_values = [3] # only used to label some of the output files. It DOES NOT have any impact in the execution of the Method

    #methods = ["NCFwR:number_random_partitions=80"]#, "NCFwR:number_random_partitions=20"],
               #"NCFwR:number_random_partitions=30", "NCFwR:number_random_partitions=60"]#,
               #"NCFwR:number_random_partitions=80", "NCFwR:number_random_partitions=100"]
    methods = ["NCFwR:number_random_partitions=80"]

    datasources = ["BBCseg3"]#BBCseg2,"BBCseg3"]#"Handwritten"]#,"NusWide"]#,"BBCseg4","Caltech20","Reuters"]#,"Handwritten","NusWide"]

    # each run will receive the following parameter configuration: {param0:k-value , param1:method-indicator, param2:dataset-name}
    # see @HDF5Datasets for the available dataset handlers.
    computation_lst = list(product(*[k_values, methods, datasources]))

    start_time = perf_counter()
    logger.debug("Starting parallel procedure...")

    #np.random.seed(initial_seed)

    with Pool(os.cpu_count(), initializer, (datapath, initial_seed, nruns)) as p: # initializer and arguments called by each process.
        result_lst = p.map(perform_single_run, computation_lst)

    # storage routines
    end_time = perf_counter()
    elapsed = ceil(end_time - start_time)
    logger.info("Overall procedure ended.")

    try:
        os.mkdir('stage1_results')
    except OSError as error:
        logger.warn('Directory stage1_results exists.')

    now = datetime.now()
    outputfile = "stage1_results/RES_{0}_{1}secs.json".format(now.strftime("%b%d%Y.%H%M%S"), elapsed)

    ## backup in case something fails below
    with open('{0}.backup.pickle'.format(outputfile), 'wb') as f:
        pickle.dump(result_lst, f, protocol=2)
        logger.info("Backup pickle dump written to {0}".format('{0}.backup.pickle'.format(outputfile)))

    #result_lst = pickle.load(open('RES_Nov062020.094810_284secs.json.backup.pickle', 'rb'))
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

        #logger.debug('--> ds_dict datasrc:{0} struct:{1}'.format(datasrc, ds_dict[datasrc]))
        for v in ds_dict[datasrc]:
            if 'K' in ds_dict[datasrc][v] and isinstance(ds_dict[datasrc][v]['K'], np.ndarray):
                ds_dict[datasrc][v]['K'] = (ds_dict[datasrc][v]['K']).tolist()
            #logger.debug("list of k:{0} --> {1}".format(ds_dict[datasrc][v]['K'], type(ds_dict[datasrc][v]['K'] )) )
        #ds_dict[datasrc]['K'] = list(ds_dict[datasrc]['K'])
        results[method][datasrc] = ds_dict[datasrc]

    logger.info("Elapsed time {0:.3f} secs".format(end_time - start_time))


    with open(outputfile, 'w') as fp:
        json.dump(results, fp)
    logger.info("Results stored into file {0}".format(outputfile))
