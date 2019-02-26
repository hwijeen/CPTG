import logging
from setproctitle import setproctitle

from dataloading import build_data
from model import make_model


DATA_DIR = '/home/nlpgpu5/hwijeen/CPTG/data/yelp/'


setproctitle("(hwijeen) CPTG in progress")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":

    data = build_data(DATA_DIR)
    cptg = make_model(len(data.vocab), len(data.attr)) 

    print(data.pos.train[0].sent)
    print(cptg)

