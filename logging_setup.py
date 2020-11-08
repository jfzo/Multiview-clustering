import logging
from datetime import datetime

logger = logging.getLogger('NCF clustering')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
now = datetime.now()
outputfile = "logging-output2/LOG_{0}.log".format(now.strftime("%b%d%Y.%H%M%S"))
fh = logging.FileHandler(outputfile, mode='w')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(filename)s(%(lineno)d):%(funcName)s / %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

 
