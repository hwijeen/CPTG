# model from the paper `Content Preserving Text Generation with Attribute Controls'

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical

# TODO: generation max length from shen et al.
MAXLEN = 30
SOS_IDX = 0 # FIXME: check this!
EOS_IDX = 1


# TODO: dropout?
class CPTG(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x, l, l_):
        """

        """
        hx, hy, gen_output = self.generator(x, l, l_)
        dis_output = self.discriminator(hx, hy, l, l_)
        return gen_output, dis_output # tuple (B, MAXLEN, 700), (B,) and
                                      # 


class Generator(nn.Module):
    def __init__(self, encoder, decoder, gamma):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.gamma = gamma

    def _fuse(self, z_x, z_y):
        B = z_x.size(0)
        g = torch.empty_like(z_x).bernoulli_(self.gamma) # (B, 500)
        z_xy = (g * z_x) - ((g != 1) * z_y)
        z_xy = (g * z_x) - ((torch.ones_like(g) - g) * z_y)
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
        _, z_x = nn.GRU(packed_in)
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
        prob = F.softmax(output.squeeze(1))
        sampled = torch.multinomial(prob, num_samples=1)
        len_ = (sampled != EOS_IDX).squeeze(1)
        return sampled.detach(), len_ # (B, 1), (B,)

    def forward(self, z, l, x=None):
        """
        z: (B, 500)
        attr: (B,)
        x: tuple of (B, L), (B,)
        """
        B = l.size(0)
        l_embed = self.attr_emb(l) # (B, 200)
        init_hidden = torch.cat([z, l_embed], dim=-1).unsqueeze(0) # (1, B, 700)
        if x is not None: # for loss computation
            x, lengths = x
            x_embed = self.emb(x) # (B, L, 300)
            packed_in = pack_padded_sequence(x_embed, lengths, batch_first=True)
            packed_out, _ = self.GRU(packed_in, init_hidden)
            total_length = x.size(1)
            # (B, L, 700)
            hx, lengths = pad_packed_sequence(packed_out, batch_first=True,
                                                  total_length=total_length)
            output = self.out(output)
            return (hx, lengths), (output, lengths) # ((B, L, 700), (B,))
                                                    # ((B, L, vocab), (B,))

        else: # sample y
            y = []
            hy = []
            input_ = z.new_full((B, 1), SOS_IDX).long()
            lengths = input_.new_ones(B)
            for t in range(MAXLEN):
                input_ = self.emb(input_) # (B, 1, 300)
                # output (B, 1, 700), hidden (1, B, 700)
                output, hidden = self.GRU(input, hidden)
                input_, len_ = _hard_sampling(self.out(output))
                hy.append(output)
                y.append(input_)
                lengths += len_
            hy = torch.cat(output, dim=1)
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
    def __init__(self):
        super().__init__(attr)
        # FIXME: just rnn?
        self.phi = nn.rnn(700, 500, batch_first=True, bidirectinoal=True)
        self.W = nn.Linear(500 * 2, attr)
        self.v = nn.Parameter(torch.randn(500 * 2))

    def _get_last_hidden(self, h):
        h, lengths = h
        B, total_lengths, _ = h.size()
        packed_in = pack_padded_sequence(h, lengths, batch_fist = true)
        packed_out, _ = self.phi(packed_in)
        # (B, L, 2*500)
        output = pad_packed_sequence(packed_out, batch_first=True,
                                     total_length=total_length)
        # (B, 2*500) 
        last_hidden = output[range(B), lengths]
        return last_hidden

    def _discriminator(self, h, l):
        last_hidden = self._get_last_hidden(h)
        term1 = torch.dot(l, )
        pass

    # FIXME: is l 'binary vector'?
    def forward(self, hx, hy, l, l_):
        """
        hx: tuple of (B, L, 700), (B,)
        hy: tuple of (B, MAXLEN, 700), (B,)
        l: (B, )
        """
        self._discriminator(hy, l_)
        self._discriminator(hx, l)
        self._discriminator(hx, l_) # FIXME: clone needed?
