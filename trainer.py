import logging

import torch
import torch.nn as nn
import torch.optim as optim

from utils import prepare_batch, truncate

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, model, data, lambda_=0.5):
        self.model = model
        self.data = data
        self.lambda_ = lambda_
        # FIXME: does reduction='mean' consider ingnore_index?
        self.g_criterion = nn.CrossEntropyLoss(ignore_index=1, reduction='none') # PAD
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

        # ganhack
        #real_loss.backward(retrain_graph=True)
        #fake_loss.backward()
        # FIXME: is this correct?
        adv_loss = real_loss + fake_loss

        adv_loss.backward(retain_graph=True)
        self.d_optimizer.step()
        self.d_optimizer.zero_grad()
        return adv_loss

    def _generator_step(self, gen_logit, x):
        logit, _ = gen_logit # (B, L+2, vocab), (B,)
        target, lengths = x # (B, L+2)
        #target = truncate(target, 'sos') # (B,
        target = torch.cat([t.view(-1) for t in target], dim=0)
        num_token = torch.sum(lengths).float()

        B, L, _ = logit.size()
        recon_loss = torch.sum(self.g_criterion(logit.view(B*L, -1),
                                          target.view(-1))) / num_token

        recon_loss.backward()
        self.g_optimizer.step()
        self.g_optimizer.zero_grad()
        return recon_loss

    # FIXME: careful - SOS sentence EOS
    # TODO: how to calculate and update with adversarial loss?
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

    def evaluate(self):
        raise NotImplementedError

    def inference(self, pos_test_iter, neg_test_iter):
        pass
