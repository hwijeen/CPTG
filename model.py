# model from the paper `Content Preserving Text Generation with Attribute Controls`

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import make_one_hot


# TODO: generation max length from shen et al.
MAXLEN = 30
SOS_IDX = 0 # FIXME: check this!
EOS_IDX = 1
GAMMA = 0.5
LAMBDA = 0.5


# TODO: dropout?
class CPTG(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x, l, l_):
        """
        x: tuple of (B, L) and (B,)
        l: (B, )
        l_: (B, )
        """
        hx, hy, gen_output = self.generator(x, l, l_)
        dis_output = self.discriminator(hx, hy, l, l_)
        return gen_output, dis_output # tuple (B, MAXLEN, 700), (B,) and
                                      # tuple of hx_l, hy_l_, hx_l_


class Generator(nn.Module):
    def __init__(self, encoder, decoder, gamma):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.gamma = gamma

    def _fuse(self, z_x, z_y):
        B = z_x.size(0)
        g = torch.empty_like(z_x).bernoulli_(self.gamma) # (B, 500)
        z_xy = (g * z_x) - ((g != 1).float() * z_y)
        return z_xy # (B, 500)

    def forward(self, x, l, l_):
        """
        x: tuple of (B, L) and (B,)
        l: (B, )
        l_: (B, )
        """
        z_x = self.encoder(x)
        hy, y = self.decoder(z_x, l_)
        z_y = self.encoder(y)
        z_xy = self._fuse(z_x, z_y)
        hx, output = self.decoder(z_xy, l, x)
        return hx, hy, output


class Encoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        # TODO: use pretrained GLOVE
        self.word_emb = nn.Embedding(vocab, 300)
        self.GRU = nn.GRU(300, 500, batch_first=True)

    def forward(self, x):
        """
        x: tuple of (B,L) and (B,)

        """
        x, lengths = x
        x_embed = self.word_emb(x) #(B, L, 300)
        packed_in = pack_padded_sequence(x_embed, lengths, batch_first=True)
        _, z_x = self.GRU(packed_in)
        return z_x.squeeze() # (B, 500)


class Decoder(nn.Module):
    def __init__(self, vocab, attr):
        super().__init__()
        self.emb = nn.Embedding(vocab, 300) # FIXME: sharing?
        self.attr_emb = nn.Embedding(attr, 200)
        self.GRU = nn.GRU(300, 700, batch_first=True)
        self.out = nn.Linear(700, vocab)

    # this does not backpropagate at all
    def _hard_sampling(self, output):
        # output (B, 1, vocab)
        prob = F.softmax(output.squeeze(1), dim=-1)
        sampled = torch.multinomial(prob, num_samples=1)
        len_ = (sampled != EOS_IDX).squeeze(1).long()
        return sampled.detach(), len_ # (B, 1), (B,)

    def forward(self, z, l, x=None):
        """
        z: (B, 500)
        attr: (B,)
        x: tuple of (B, L), (B,)
        l: (B,)
        """
        B = l.size(0)
        l_embed = self.attr_emb(l) # (B, 200)
        hidden = torch.cat([z, l_embed], dim=-1).unsqueeze(0) # (1, B, 700)

        if x is not None: # for loss computation
            x, lengths = x
            x_embed = self.emb(x) # (B, L, 300)
            packed_in = pack_padded_sequence(x_embed, lengths, batch_first=True)
            packed_out, _ = self.GRU(packed_in, hidden)
            total_length = x.size(1)
            # (B, L, 700)
            hx, lengths = pad_packed_sequence(packed_out, batch_first=True,
                                                  total_length=total_length)
            output = self.out(hx)
            return (hx, lengths), (output, lengths) # ((B, L, 700), (B,))
                                                    # ((B, L, vocab), (B,))

        else: # sample y
            y = []
            hy = []
            input_ = z.new_full((B, 1), SOS_IDX).long()
            lengths = input_.new_zeros(B)
            for t in range(MAXLEN):
                input_ = self.emb(input_) # (B, 1, 300)
                # output (B, 1, 700), hidden (1, B, 700)
                output, hidden = self.GRU(input_, hidden)
                input_, len_ = self._hard_sampling(self.out(output))
                hy.append(output)
                y.append(input_)
                lengths += len_
            hy = torch.cat(hy, dim=1)
            y = torch.cat(y, dim=1)
            # y = self._tighten(y, lengths)
            return (hy, lengths), (y, lengths) # (B, MAXLEN, 700), (B, MAXLEN), (B, )

        # TODO: tighten sampled batch
        def _tighten(self, y, lengths):
            """
            truncate tokens after EOS
            """
            pass


class Discriminator(nn.Module):
    def __init__(self, attr):
        super().__init__()
        # FIXME: just rnn?
        self.attr = attr
        self.birnn = nn.RNN(700, 500, batch_first=True, bidirectional=True)
        self.W = nn.Linear(500 * 2, attr)
        self.v = nn.Parameter(torch.randn(500 * 2))

    def _phi(self, h):
        """
        h: tuple of (B, L, 700) and (B,)
        """
        h, lengths = h
        B, total_length, _ = h.size()
        packed_in = pack_padded_sequence(h, lengths, batch_first=True)
        packed_out, _ = self.birnn(packed_in)
        # (B, L, 2*500)
        output, lengths = pad_packed_sequence(packed_out, batch_first=True,
                                     total_length=total_length)
        ## (B, 2*500) 
        ##last_hidden = output[range(B), lengths]
        output = output.view(B, total_length, 2, -1 )
        forward = output[:, :, 0]
        h_0 = forward[:, 0]
        backward = output[:, :, 1]
        h_t = backward[range(B), lengths-1] # index starts from 0
        last_hidden = torch.cat([h_0, h_t], dim=1)
        return last_hidden # (B, 500*2)

    def _discriminator(self, h, l):
        last_hidden = self._phi(h)
        # FIXME: is l 'binary vector'?
        l_onehot = make_one_hot(l, self.attr)
        #term1 = torch.matmul(l_onehot, self.W(last_hidden).t()) # batch dot product
        #term2 = torch.matmul(self.v, last_hidden.t())
        term1 = torch.sum(l_onehot * self.W(last_hidden), dim=1)
        term2 = torch.sum(self.v * last_hidden, dim=1)
        return torch.sigmoid(term1 + term2)

    def forward(self, hx, hy, l, l_):
        """
        hx: tuple of (B, L, 700), (B,)
        hy: tuple of (B, MAXLEN, 700), (B,)
        l: (B, )
        """
        # TODO: ground truth answer for these
        hx_l = self._discriminator(hx, l)
        hy_l_ = self._discriminator(hy, l_)
        hx_l_ = self._discriminator(hx, l_) # FIXME: clone needed?
        return hx_l, hy_l_, hx_l_


def make_model(vocab, attr):
    encoder = Encoder(vocab)
    decoder = Decoder(vocab, attr)
    generator = Generator(encoder, decoder, GAMMA)
    discriminator = Discriminator(attr)
    return CPTG(generator, discriminator)

if __name__ == "__main__":
    vocab = 20000
    attr = 2
    B = 32
    L = 20

    cptg = make_model(vocab, attr)

    x = torch.randint(0, vocab, (B, L)).long()
    length, _ = torch.randint(1, L, (B,)).long().sort(descending=True)
    input_ = (x, length)
    l = torch.empty(B, ).bernoulli_(0.5).long()
    l_ = (l != 1).long()

    gen_output, dis_output = cptg(input_, l, l_)

    x_logits = gen_output[0]
    hx_l, hy_l_, hx_l_ = dis_output

    loss = torch.sum(x_logits) + torch.sum(hx_l + hy_l_ + hx_l_)
    print('loss: ', loss.item())

    loss.backward()

    cnt = 0
    for name, param in cptg.named_parameters():
        if param.grad is None:
            print('no grad in {}'.format(param))
        else:
            cnt += 1
    assert len(list(cptg.parameters())) == cnt, 'gradient is not backproped somewhere'
