import torch

from dataloading import EOS_IDX


def make_one_hot(l, attr):
    B = l.size(0)
    one_hot = l.new_zeros(B, attr).float()
    one_hot[range(B), l] = 1
    return one_hot

def prepare_batch(batch, label):
    # attach label and unpack 'batch'
    x, lengths = batch.sent
    B = x.size(0)
    if label == 'pos':
        l = x.new_ones(B,)
        l_ = x.new_zeros(B,)
    elif label == 'neg':
        l = x.new_zeros(B,)
        l_ = x.new_ones(B,)
    return (x, lengths), l, l_

def truncate(x, token=None):
    assert token in ['sos', 'eos', 'both'], 'can only truncate sos or eos'
    x, lengths = x # (B, L)
    lengths -= 1
    if token == 'sos': x = x[:, 1:]
    elif token == 'eos': x = x[:, :-1]
    else: x = x[:, 1:-1]
    return (x, lengths)

def append(x, token=None):
    assert token in ['sos', 'eos'], 'can only truncate sos or eos'
    x, lengths = x # (B, L)
    B = x.length(0)
    lengths += 1
    if token == 'eos':
        eos = x.new_full((B,1), EOS_IDX)
        x = torch.cat([x, eos], dim=1)
    elif token == 'sos':
        raise NotImplementedError
    return (x, lengths)


