import torch
import torch.nn as nn
import torch.optim as optim

from utils import prepare_batch, truncate

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

    def _discriminator_step(self, pos_dis_logit, neg_dis_logit):
        pos_hx_l, pos_hy_l_, pos_hx_l_ = pos_dis_logit
        neg_hx_l, neg_hy_l_, neg_hy_l_ = neg_dis_logit
        B = pos_hx_l.size(0)
        real_label = pos_hx_l.new_ones(B,)
        fake_label = pos_hx_l.new_zeros(B,)

        pos_real_loss = self.lambda_ * 2 * self.d_criterion(pos_hx_l, real_label)
        pos_fake_loss = self.lambda_ * (self.d_criterion(pos_hy_l_, fake_label) +\
            self.d_criterion(pos_hx_l_, fake_label))

        neg_real_loss = self.lambda_ * 2 * self.d_criterion(neg_hx_l, real_label)
        neg_fake_loss = self.lambda_ * (self.d_criterion(neg_hy_l_, fake_label) +\
            self.d_criterion(neg_hy_l_, fake_label))

        # ganhack
        #real_loss = pos_real_loss + neg_real_loss
        #fake_loss = pos_fake_loss + neg_fake_loss
        #real_loss.backward()
        #fake_loss.backward()
        # FIXME: is this correct?
        adv_loss = pos_real_loss + neg_real_loss + pos_fake_loss + neg_fake_loss
        adv_loss.backward()
        self.d_optimizer.step()
        self.d_optimizer.zero_grad()

    def _generator_step(self, pos_gen_logit, neg_gen_logit, pos_batch, neg_batch):
        pos_logit, _ = pos_gen_logit # (B, L+2, vocab), (B,)
        neg_logit, _ = neg_gen_logit
        pos_target, pos_lengths = pos_batch # (B, L+2)
        neg_target, neg_lengths = neg_batch
        #pos_target = truncate(pos_target, 'sos') # (B,
        #neg_target = truncate(neg_target, 'sos')
        pos_total_token = torch.sum(pos_lengths)
        neg_total_token = torch.sum(neg_lengths)

        B_pos, L_pos, _ = pos_logit.size()
        B_neg, L_neg, _ = neg_logit.size()
        pos_recon_loss = torch.sum(self.g_criterion(pos_logit.view(B_pos*L_pos, -1),
                                          pos_target.view(-1))) / pos_total_token
        neg_recon_loss = torch.sum(self.g_criterion(neg_logit.view(B_neg*L_neg, -1),
                                   neg_target.view(-1))) / neg_total_token
        recon_loss = pos_recon_loss + neg_recon_loss
        recon_loss.backward()

        self.g_optimizer.step()
        self.g_optimizer.zero_grad()

    # FIXME: careful - SOS sentence EOS
    # TODO: how to calculate and update with adversarial loss?
    def train(self, epoch):
        for i in range(epoch):
            # FIXME: pos and dev data have different size
            for pos_batch, neg_batch in zip(self.data.pos.train_iter, self.data.neg.train_iter):
            #for pos_batch, neg_batch in zip(pos_train_iter, neg_train_iter):
                pos_batch = prepare_batch(pos_batch, 'pos')
                neg_batch = prepare_batch(neg_batch, 'neg')
                pos_gen_logit, pos_dis_logit = self.model(*pos_batch)
                neg_gen_logit, neg_dis_logit = self.model(*neg_batch)
                self._discriminator_step(pos_dis_logit, neg_dis_logit)
                self._generator_step(pos_gen_logit, neg_gen_logit,
                                     pos_batch, neg_batch)

    def evaluate(self):
        raise NotImplementedError

    def inference(self, pos_test_iter, neg_test_iter):
        pass
