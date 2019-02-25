import logging
from setproctitle import setproctitle

from dataloading import Data
from model import CPTG


DATA_DIR = '/home/nlpgpu5/hwijeen/CPTG/data/yelp/'


setproctitle("(hwijeen) CPTG in progress")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":

    logger.info('loaded data from... {}, label...pos'.format(DATA_DIR))
    pos_data = Data(DATA_DIR, 'pos')
    logger.info('loaded data from... {}, label...neg'.format(DATA_DIR))
    neg_data = Data(DATA_DIR, 'neg')

