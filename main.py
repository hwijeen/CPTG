import os
import logging
from setproctitle import setproctitle

import torch
import torch.nn as nn

from dataloading import build_data
from model import make_model
from utils import prepare_batch # temp
from trainer import Trainer


DATA_DIR = '/home/nlpgpu5/hwijeen/CPTG/data/yelp/'
# FIXME: thorough device control
device = torch.device('cuda:0')
# TODO: multi-gpu
multi_gpu = True


setproctitle("(hwijeen) CPTG in progress")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":

    data = build_data(DATA_DIR, batch_size=32, device=device)
    cptg = make_model(len(data.vocab), len(data.attr), device=device)
    trainer = Trainer(cptg, data)
    print(cptg)

    trainer.train(epoch=5)



