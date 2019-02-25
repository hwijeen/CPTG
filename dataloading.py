import os
import torch
import logging
from torchtext.data import Field, TabularDataset, BucketIterator


MAXLEN = 15
MAXVOCAB = 30000
logger = logging.getLogger(__name__)


class Data(object):
    def __init__(self, data_dir, label):
        self.label = label
        if label == 'pos':
            self.train_path = os.path.join(data_dir, 'sentiment.train.1')
            self.val_path = os.path.join(data_dir, 'sentiment.dev.1')
            self.test_path = os.path.join(data_dir, 'sentiment.test.1')
        elif label == 'neg':
            self.train_path = os.path.join(data_dir, 'sentiment.train.0')
            self.val_path = os.path.join(data_dir, 'sentiment.dev.0')
            self.test_path = os.path.join(data_dir, 'sentiment.test.0')
        self.build()

    def build(self):
        self.sent_field = self.build_field(maxlen=MAXLEN)
        logger.info('building datasets...{}'.format(self.label))
        self.train, self.val, self.test = self.build_dataset(self.sent_field)
        self.vocab = self.build_vocab(self.sent_field, self.train.sent,
                                      self.train.sent)
        self.train_iter, self.valid_iter, self.test_iter =\
            self.build_iterator(self.train, self.val, self.test)
        logger.info('data size... {} / {} / {}'.format(len(self.train),
                                                       len(self.val),
                                                       len(self.test)))
        logger.info('vocab size... {}'.format(len(self.vocab)))

    def build_field(self, maxlen=None):
        src_field= Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1])
        return src_field

    def build_dataset(self, field):
        train = TabularDataset(path=self.train_path, format='tsv',
                               fields=[('sent', field)])
        val = TabularDataset(path=self.val_path, format='tsv',
                               fields=[('sent', field)])
        test = TabularDataset(path=self.test_path, format='tsv',
                               fields=[('sent', field)])
        return train, val, test

    def build_vocab(self, field, *args):
        # not using pretrained word vectors
        field.build_vocab(args, max_size=MAXVOCAB)
        return field.vocab

    def build_iterator(self, train, val, test):
        train_iter, valid_iter, test_iter = \
        BucketIterator.splits((train, val, test), batch_size=32,
                              sort_key=lambda x: len(x.sent),
                              sort_within_batch=True, repeat=False,
                              device=torch.device('cuda'))
        return train_iter, valid_iter, test_iter



