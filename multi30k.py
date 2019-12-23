from __future__ import division
from __future__ import print_function

import os
import numpy as np
from glob import glob
from data_loader import DatasetLoader

class Multi30KLoader(DatasetLoader):
    """ Dataset loader class that loads feature matrices from given paths and
        create shuffled batch for training, unshuffled batch for evaluation.
    """
    def get_tokens(self, args, language):
        token_folder = os.path.join('data', args.dataset, 'tokenized', language)
        if self.split == 'train':
            # training set also includes translations with a different prefix
            token_filenames = glob(os.path.join(token_folder, '*' + self.split + '*'))
        else:
            token_filenames = glob(os.path.join(token_folder, self.split + '*'))

        tokens = [[] for _ in range(len(self.image2index))]
        vocab = set()
        max_length = 0
        for token_name in token_filenames:
            sentences = open(token_name, 'r')
            sentences = sentences.readlines()
            for i, sentence in enumerate(sentences):
                sentence = sentence.lower().split()
                vocab.update(sentence)
                max_length = max(len(sentence), max_length)
                tokens[i].append(sentence)

        im2sent = {}
        sent2im = []
        num_sentences = 0
        for i, sentences in enumerate(tokens):
            im2sent[i] = np.arange(num_sentences, num_sentences + len(sentences))
            sent2im.append(np.ones(len(sentences), np.int32) * i)
            num_sentences += len(sentences)

        sent2im = np.hstack(sent2im)
        max_length = min(max_length, args.max_sentence_length)
        return tokens, sent2im, im2sent, vocab, max_length
