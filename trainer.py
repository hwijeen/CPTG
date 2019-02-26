from utils import attach_label

class Trainer(object):
    def __init__(self, model, data, criterion, g_optimizer, d_optimizer):
        self.model = model
        self.data = data
        self.criterion = criterion
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    # FIXME: careful - SOS sentence EOS
    # TODO: how to calculate and update with adversarial loss?
    def run_epoch(pos_train_iter, neg_train_iter):
        for pos_batch, neg_batch in zip(pos_train_iter, neg_train_iter):
            pos_batch = attach_label(pos_batch, 'pos')
            neg_batch = attach_label(neg_batch, 'neg')
            pos_gen_logit, pos_dis_logit = self.model(pos_batch)
            neg_gen_logit, neg_dis_logit = self.model(neg_batch)

    def evaluate():
        raise NotImplementedError

    def inference(pos_test_iter, neg_test_iter):
        pass
