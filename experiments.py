# 20-07-2020
from original_ncf import NCF
import utils
import logging
from logging_setup import logger

from data_source import TwentyNewsgroupView, BBCSportsView, ReutersView, WEBKBView
 

if __name__ == "__main__":
    #logger = logging.getLogger(__name__)
    seed = 123
    twview_gen = TwentyNewsgroupView("../../data", 5, seed)
    bbcview_gen = BBCSportsView("../../data", 5, seed)
    reutersview_gen = ReutersView("../../data", 5, seed)
    webkbview_gen = WEBKBView("../../data", 5, seed)

    for ds in [twview_gen, bbcview_gen, reutersview_gen, webkbview_gen]:
        ncf_method = NCF(ds.get_views() )
        ncf_consensus = ncf_method.run()
        logger.debug("Performance of %s" % (ds.name) )
        logger.debug("Entropy: %f" % (utils.Entropy(ncf_consensus, ds.get_real_labels() )) )
        logger.debug("Purity: %f" % (utils.Purity(ncf_consensus, ds.get_real_labels())) )