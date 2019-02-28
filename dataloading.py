import os
import logging

import torch
from torchtext.data import RawField, Field, Dataset, TabularDataset, BucketIterator


MAXLEN = 15
MAXVOCAB = 30000
logger = logging.getLogger(__name__)

# set by torchtext
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
POS_LABEL = 1
NEG_LABEL = 0

class PosNegData(object):
    def __init__(self, pos, neg, batch_size=32):
        self.sent_field = pos.sent_field
        self.train = self.merge_data(pos.train, neg.train)
        self.valid = self.merge_data(pos.valid, neg.valid)
        self.test = self.merge_data(pos.test, neg.test)
        self.attr = [pos.label, neg.label]
        self.vocab = self.build_vocab()
        self.train_iter, self.valid_iter, self.test_iter =\
            self.build_iterator(batch_size)

    def _attach_label(self, ex, label):
        setattr(ex, 'label', label)
        return ex

    def merge_data(self, pos, neg):
        label_field = RawField(postprocessing=lambda x: torch.cuda.LongTensor(x))
        label_field.is_target = True
        examples = [self._attach_label(ex, POS_LABEL) for ex in pos] +\
            [self._attach_label(ex, NEG_LABEL) for ex in neg]
        return Dataset(examples, [('sent', self.sent_field),
                                  ('label', label_field)])

    def build_vocab(self, pretrained=None):
        # not using pretrained word vectors
        self.sent_field.build_vocab(self.train, self.valid, max_size=MAXVOCAB)
        return self.sent_field.vocab

    def build_iterator(self, batch_size):
        train_iter, valid_iter, test_iter = \
        BucketIterator.splits((self.train, self.valid, self.test),
                              batch_size=batch_size,
                              sort_key=lambda x: len(x.sent),
                              sort_within_batch=True, repeat=False,
                              device=torch.device('cuda'))
        return train_iter, valid_iter, test_iter


class LabeledData(object):
    def __init__(self, data_dir, label):
        self.label = label
        if label == 'pos':
            self.train_path = os.path.join(data_dir, 'sentiment.train.1')
            self.valid_path = os.path.join(data_dir, 'sentiment.dev.1')
            self.test_path = os.path.join(data_dir, 'sentiment.test.1')
        elif label == 'neg':
            self.train_path = os.path.join(data_dir, 'sentiment.train.0')
            self.valid_path = os.path.join(data_dir, 'sentiment.dev.0')
            self.test_path = os.path.join(data_dir, 'sentiment.test.0')
        self.build()

    def build(self):
        self.sent_field = self.build_field(maxlen=MAXLEN)
        self.train, self.valid, self.test = self.build_dataset(self.sent_field)

    def build_field(self, maxlen=None):
        sent_field= Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        init_token='<sos>', eos_token='<eos>')
        return sent_field

    def build_dataset(self, field):
        train = TabularDataset(path=self.train_path, format='tsv',
                               fields=[('sent', field)])
        valid = TabularDataset(path=self.valid_path, format='tsv',
                               fields=[('sent', field)])
        test = TabularDataset(path=self.test_path, format='tsv',
                               fields=[('sent', field)])
        return train, valid, test


def build_data(data_dir, batch_size):
    logger.info('loading data from... {}, label...pos'.format(data_dir))
    pos_data = LabeledData(data_dir, 'pos')
    logger.info('loading data from... {}, label...neg'.format(data_dir))
    neg_data = LabeledData(data_dir, 'neg')

    data = PosNegData(pos_data, neg_data, batch_size)
    logger.info('total dataset size... {} / {} / {}'.format(
    len(data.train), len(data.valid), len(data.test)))
    logger.info('vocab size... {}'.format(len(data.vocab)))
    return data

if __name__ == "__main__":
    DATA_DIR = '/home/nlpgpu5/hwijeen/CPTG/data/yelp/'
    data = build_data(DATA_DIR, 32)
    print(len(data.train), len(data.valid), len(data.test))
    print(len(data.vocab))
    for batch in data.train_iter:
        print(batch)
        print(batch.sent)
        print(batch.label.size())
        input()
