import logging

import torch
import torch.nn as nn
import torch.optim as optim

from dataloading import PAD_IDX
from utils import prepare_batch, reverse

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, model, data, lambda_=0.5):
        self.model = model
        self.data = data
        self.lambda_ = lambda_
        # FIXME: does reduction='mean' consider ingnore_index?
        self.g_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX,
                                               reduction='none')
        self.d_criterion = nn.BCEWithLogitsLoss()
        self.g_optimizer = optim.Adam(model.generator.parameters())
        self.d_optimizer = optim.Adam(model.discriminator.parameters())

    def _discriminator_step(self, dis_logit):
        hx_l, hy_l_, hx_l_ = dis_logit
        B = hx_l.size(0)
        real_label = hx_l.new_ones(B,)
        fake_label = hx_l.new_zeros(B,)

        real_loss = self.lambda_ * 2 * self.d_criterion(hx_l, real_label)
        fake_loss = self.lambda_ * (self.d_criterion(hy_l_, fake_label) +\
            self.d_criterion(hx_l_, fake_label))

        adv_loss = real_loss + fake_loss

        adv_loss.backward(retain_graph=True)
        self.d_optimizer.step()
        self.d_optimizer.zero_grad()
        return adv_loss

    def _generator_step(self, gen_logit, x):
        logit, _ = gen_logit # (B, L+1, vocab), (B,)
        target, lengths = x # (B, L+1)
        target = torch.cat([t.view(-1) for t in target], dim=0)
        num_token = torch.sum(lengths).float()

        B, L, _ = logit.size()
        recon_loss = torch.sum(self.g_criterion(logit.view(B*L, -1),
                                          target.view(-1))) / num_token

        recon_loss.backward()
        self.g_optimizer.step()
        self.g_optimizer.zero_grad()
        return recon_loss

    # TODO: logging and tqdm
    def train(self, epoch):
        for i in range(epoch):
            for step, batch in enumerate(self.data.train_iter):
                (x, lengths), l, l_ = prepare_batch(batch)
                gen_logit, dis_logit = self.model((x, lengths), l, l_)
                adv_loss = self._discriminator_step(dis_logit)
                recon_loss = self._generator_step(gen_logit, (x, lengths))

                if step % 100 == 0:
                    logger.info('loss at epoch {}, step {}: {:.2f}'.format(
                    i, step, self.lambda_ * adv_loss + recon_loss))
            # TODO: implement evaluation(inference)
                if step % 1000 == 0:
                    self.evaluate()

    # TODO: early stopping
    def evaluate(self):
        self.data.valid_iter.shuffle = True 
        import random
        a = random.randint(0, len(self.data.valid_iter)) # temp test
        for i, batch in enumerate(self.data.valid_iter):
            if i != a: continue
            # FIXME: feed l when inferring?
            (x, lengths), l, l_ = prepare_batch(batch)
            generated = self.model((x, lengths), l, l_, is_gen=True)
            #decoded = self.decode(gen_logit)
            print('=' * 50)
            print('original \t\t -> \t\t changed')
            for idx in random.sample(range(lengths.size(0)), 5):
                ori = reverse(x, self.data.vocab)[idx]
                chg = reverse(generated[0], self.data.vocab)[idx]
                print(' '.join(ori))
                print('\t\t->', ' '.join(chg))
            #print('\n'.join([' '.join(ex) for ex in
            #                 random.sample(reverse(x, self.data.vocab), 3)]))
            #print('\n'.join([' '.join(ex) for ex in
            #                random.sample(reverse(generated[0], self.data.vocab), 3)]))
            print('=' * 50)
            return


    #def decode(self, gen_logit):
        #pass

    #def inference(self, pos_test_iter, neg_test_iter):
    #    pass
