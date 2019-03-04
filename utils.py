import torch

from dataloading import EOS_IDX, SOS_IDX


def make_one_hot(l, attr):
    B = l.size(0)
    one_hot = l.new_zeros(B, attr).float()
    one_hot[range(B), l] = 1
    return one_hot

def prepare_batch(batch):
    # attach the opposite label
    x, lengths = batch.sent
    l = batch.label
    B = x.size(0)
    l_ = (l != 1).long()
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
    assert token in ['sos', 'eos'], 'can only append sos or eos'
    x, lengths = x # (B, L)
    B = x.size(0)
    lengths += 1
    if token == 'eos':
        eos = x.new_full((B,1), EOS_IDX)
        x = torch.cat([x, eos], dim=1)
    elif token == 'sos':
        sos = x.new_full((B,1), SOS_IDX)
        x = torch.cat([sos, x], dim=1)
    return (x, lengths)

def reverse(batch, vocab):
    batch = batch.tolist()

    def trim(s, t):
        sentence = []
        for w in s:
            if w == t:
                break
            sentence.append(w)
        return sentence
    batch = [trim(ex, EOS_IDX) for ex in batch]

    batch = [[vocab.itos[i] for i in ex] for ex in batch]
    return batch



