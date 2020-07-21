# 20-07-2020
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
    #initial_seed = 123
    initial_seed = 110
    nruns = 3
    data = "D:\Google Drive\Research - Multiview and Collaborative Clustering\data"
    header = ["method", "k", "dataset", "entropy", "purity"]
    tdata = []

    k_val = 5
    start_time = perf_counter()

    results = {}
    for met_name, met_op in zip(["NCF","NCFwR"],[NCF, NCFwR]):
        results[met_name] = {}
        for r in range(nruns):
            seed = np.random.randint(1, 1e5)

            twview_gen = TwentyNewsgroupView(data, k_val, seed)
            bbcview_gen = BBCSportsView(data, k_val, seed)
            reutersview_gen = ReutersView(data, k_val, seed)
            webkbview_gen = WEBKBView(data, k_val, seed)

            # each iteration results in performance measures.
            for ds in [twview_gen, bbcview_gen, reutersview_gen, webkbview_gen]:
                ncf_method = met_op(ds.get_views())
                ncf_consensus = ncf_method.run()

                utils_entropy = utils.Entropy(ncf_consensus, ds.get_real_labels())
                utils_purity = utils.Purity(ncf_consensus, ds.get_real_labels())

                if not ds.name in results[met_name]:
                    results[met_name][ds.name] = {"entropy":[], "purity":[]}

                results[met_name][ds.name]["entropy"].append(utils_entropy)
                results[met_name][ds.name]["purity"].append(utils_purity)

                logger.debug("Performance of %s" % (ds.name) )
                logger.debug("Entropy: %f" % (utils_entropy))
                logger.debug("Purity: %f" % (utils_purity))
                #tdata.append(("NCF", 5, ds.name.replace("View",""), utils_entropy, utils_purity))
    # storage routines
    end_time = perf_counter()
    elapsed = ceil(end_time - start_time)
    now = datetime.now()
    outputfile = "RES_%s.%dsecs.json" % (now.strftime("%b%d%Y%H%M%S"), elapsed)

    logger.info("Elapsed time %.3f secs" % (end_time - start_time))
    logger.info("Results stored into file %s"%(outputfile))
    with open(outputfile, 'w') as fp:
        json.dump(results, fp)


    #print( tab(tdata, headers=header, tablefmt='fancy_grid', floatfmt='.3f') )
    #logger.info("\n%s" % tab(tdata, headers=header, tablefmt='plain', floatfmt='.3f'))