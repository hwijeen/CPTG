import os
import torch
import logging
from setproctitle import setproctitle

# FIXME: _2
from dataloading import build_data
from model import make_model
from utils import prepare_batch # temp
from trainer import Trainer


DATA_DIR = '/home/nlpgpu5/hwijeen/CPTG/data/yelp/'


setproctitle("(hwijeen) CPTG in progress")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":

    data = build_data(DATA_DIR, batch_size=2)
    cptg = make_model(len(data.vocab), len(data.attr))
    trainer = Trainer(cptg, data)
    print(cptg)

    trainer.train(epoch=2)















    ## test forward
    #for pos_batch, neg_batch in zip(data.pos.train_iter, data.neg.train_iter):
    #    pos_batch = prepare_batch(pos_batch, 'pos')
    #    neg_batch = prepare_batch(neg_batch, 'neg')
    #    pos_gen_logit, pos_dis_logit = cptg(*pos_batch)
    #    neg_gen_logit, neg_dis_logit = cptg(*neg_batch)

    #    arbitrary_loss = torch.sum(pos_gen_logit[0]) + torch.sum(neg_gen_logit[0]) +\
    #        torch.sum(sum(pos_dis_logit)) + torch.sum(sum(neg_dis_logit))
    #    arbitrary_loss.backward()
    #    break
    ## test backward
    #cnt = 0
    #for name, param in cptg.named_parameters():
    #    if param.grad is None:
    #        print('no grad in {}'.format(param))
    #    else:
    #        cnt += 1
    #assert len(list(cptg.parameters())) == cnt, 'gradient is not backproped somewhere'



