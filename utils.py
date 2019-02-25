import torch


def make_one_hot(l, attr):
    B = l.size(0)
    one_hot = torch.zeros(B, attr)
    one_hot[range(B), l] = 1
    return one_hot

def attach_label(batch, label):
    x, lengths = batch
    B = x.size(0)
    if label == 'pos':
        l = x.new_ones(B,)
        l_ = x.new_zeros(B,)
    elif label == 'neg':
        l = x.new_zeros(B,)
        l_ = x.new_ones(B,)
    return batch, l, l_
