import torch


def make_one_hot(l, attr):
    B = l.size(0)
    one_hot = torch.zeros(B, attr)
    one_hot[range(B), l] = 1
    return one_hot
