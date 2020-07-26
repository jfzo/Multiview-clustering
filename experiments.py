# 26-07-2020
# TODO: (Main) Assess the stability of the view ranks in NCFwR
# TODO: Answer how the random partitions affect the 'Bias'
# TODO: Answer how many random partitions are neccessary for the Method to obtain the same rank or performance result
# TODO: End the evaluation procedure!

from original_ncf import NCFwR, NCF
import utils
from logging_setup import logger
from tabulate import tabulate as tab
from data_source import TwentyNewsgroupView, BBCSportsView, ReutersView, WEBKBView
import numpy as np
import json
from datetime import datetime
from time import perf_counter
from math import ceil

if __name__ == "__main__":
    nruns = 3
    k_val = 5  # eventually, a different value could be specified for each different view
    initial_seed = 1101982
    # initial_seed = 123 # generates problems with a merge (ricerca!)
    n_rnd_part = 20

    # data = "D:\Google Drive\Research - Multiview and Collaborative Clustering\data"
    data = "C:/Users/juan/Insync/juan.zamora@pucv.cl/Google Drive/Research - Multiview and Collaborative Clustering/data"
    header = ["method", "k", "dataset", "entropy", "purity"]
    tdata = []
    start_time = perf_counter()
    np.random.seed(initial_seed)

    results = {}

    for r in range(nruns):
        seed = np.random.randint(1, 1e5)

        for met_name, met_op in zip(["NCF", "NCFwR"], [NCF(), NCFwR(number_random_partitions=n_rnd_part, seed=seed)]):

            if not met_name in results:
                results[met_name] = {}

            twview_gen = TwentyNewsgroupView(data, k_val, seed) # seed is employed for kmeans inside the view gen.
            bbcview_gen = BBCSportsView(data, k_val, seed)
            reutersview_gen = ReutersView(data, k_val, seed)
            webkbview_gen = WEBKBView(data, k_val, seed)

            # each iteration results in performance measures.
            for ds in [twview_gen, bbcview_gen, reutersview_gen, webkbview_gen]:
                met_op.setInputPartitions(ds.get_views())

                consensus_partition = met_op.run()
                consensus_kval = len(np.unique(consensus_partition))

                consensus_E = utils.Entropy(consensus_partition, ds.get_real_labels())
                consensus_P = utils.Purity(consensus_partition, ds.get_real_labels())

                if not ds.name in results[met_name]:
                    results[met_name][ds.name] = {}
                    for viewname in ds.data_views:
                        #if viewname in ["labels", "dataset"]:
                        #    continue
                        results[met_name][ds.name][viewname] = {"K":ds.NCLUSTERS[viewname], "entropy": [], "purity": []}
                    results[met_name][ds.name]["consensus"] = {"K":consensus_kval, "entropy": [], "purity": []}

                for viewname in ds.data_views:
                    #if viewname in ["labels", "dataset"]:
                    #    continue
                    v_E = utils.Entropy(ds.get_views()[viewname], ds.get_real_labels())
                    v_P = utils.Purity(ds.get_views()[viewname], ds.get_real_labels())
                    results[met_name][ds.name][viewname]["entropy"].append(v_E)
                    results[met_name][ds.name][viewname]["purity"].append(v_P)

                results[met_name][ds.name]["consensus"]["entropy"].append(consensus_E)
                results[met_name][ds.name]["consensus"]["purity"].append(consensus_P)

                logger.debug("Performance of %s" % (ds.name))
                logger.debug("Entropy: %f" % (consensus_E))
                logger.debug("Purity: %f" % (consensus_P))
                # tdata.append(("NCF", 5, ds.name.replace("View",""), utils_entropy, utils_purity))

    # storage routines
    end_time = perf_counter()
    elapsed = ceil(end_time - start_time)
    now = datetime.now()
    outputfile = "RES_%s_%dsecs.json" % (now.strftime("%b%d%Y.%H%M%S"), elapsed)

    logger.info("Elapsed time %.3f secs" % (end_time - start_time))
    logger.info("Results stored into file %s" % (outputfile))
    with open(outputfile, 'w') as fp:
        json.dump(results, fp)

    # print( tab(tdata, headers=header, tablefmt='fancy_grid', floatfmt='.3f') )
    # logger.info("\n%s" % tab(tdata, headers=header, tablefmt='plain', floatfmt='.3f'))
