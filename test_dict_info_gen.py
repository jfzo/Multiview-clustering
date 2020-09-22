import pickle
import numpy as np
from logging import getLogger
import json

def get_info_dict(fname):
    with open(fname, 'rb') as f:    
        result_lst = pickle.load(f)

    results = {}
    for run_ex in result_lst:
        assert len(run_ex) == 1
        #logger.debug(("KEYS in RUN {0}".format(run_ex.keys())))
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
    return results


if __name__ == '__main__':
    #fnamein = 'RES_Sep212020.180055_24120secs.json.backup.pickle'
    #R =  get_info_dict(fnamein)
    #print(R)
    logger = getLogger('test-dict-generator')

    with open('RES_Sep212020.180055_24120secs.json.backup.pickle', 'rb') as f:
        result_lst = pickle.load(f)

    results = {}
    for run_ex in result_lst:
        assert len(run_ex) == 1
        logger.debug(("KEYS in RUN {0}".format(run_ex.keys())))
        method = list(run_ex.keys())[0]
        ds_dict = run_ex[method]

        if len(ds_dict) != 1:
            continue
        datasrc = list(ds_dict.keys())[0]
        if not method in results:
            results[method] = {}
        if 'aveK_ranks' in ds_dict[datasrc]:
            ds_dict[datasrc]['aveK_ranks'] = [','.join(map(str, ranks)) for ranks in ds_dict[datasrc]['aveK_ranks']]

        # logger.debug('--> ds_dict datasrc:{0} struct:{1}'.format(datasrc, ds_dict[datasrc]))
        for v in ds_dict[datasrc]:
            if 'K' in ds_dict[datasrc][v] and isinstance(ds_dict[datasrc][v]['K'], np.ndarray):
                ds_dict[datasrc][v]['K'] = (ds_dict[datasrc][v]['K']).tolist()
            # logger.debug("list of k:{0} --> {1}".format(ds_dict[datasrc][v]['K'], type(ds_dict[datasrc][v]['K'] )) )
        # ds_dict[datasrc]['K'] = list(ds_dict[datasrc]['K'])
        results[method][datasrc] = ds_dict[datasrc]

    with open('outputfile.json', 'w') as fp:
        json.dump(results, fp)

