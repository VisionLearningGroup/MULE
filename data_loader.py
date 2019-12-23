from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle
from abc import abstractmethod

def get_sentence(vocab, tokens, token_length):
    sent_feats = []
    for sentences in tokens:
        feats = np.zeros((len(sentences), token_length), np.int32)
        for i, words in enumerate(sentences):
            words = [word for word in words if word in vocab]
            for j, word in enumerate(words[:token_length]):
                feats[i, j] = vocab[word]

        sent_feats.append(feats)

    sent_feats = np.concatenate(sent_feats, axis=0)
    return sent_feats

def get_embeddings(args, language, vocab):
    cachefn = os.path.join('data', args.dataset, language + '_vecs.pkl')
    if os.path.exists(cachefn):
        embedding_data = pickle.load(open(cachefn, 'rb'))
        word2index = embedding_data['word2index']
        vecs = embedding_data['vecs']
    else:
        embedding_dims = 300
        if language == 'cn':
            language = 'zh'
        elif language == 'jp':
            language = 'ja'

        wordvec_file = os.path.join('fasttext', 'cc.%s.300.vec' % language)
        with open(wordvec_file, 'r') as f:
            w2v_dict = {}
            for i, line in enumerate(f):
                if i % 100000 == 0:
                    print('reading %s vector %i' % (language, i))
                    
                line = line.strip()
                if not line:
                    continue

                vec = line.split()
                if len(vec) != embedding_dims + 1:
                    continue
            
                label = vec[0].lower()
                if label not in vocab:
                    continue

                vec = np.array([float(x) for x in vec[1:]], np.float32)
                assert(len(vec) == embedding_dims)
                w2v_dict[label] = vec

            vocab = vocab.intersection(set(w2v_dict.keys()))

        vocab = list(vocab)
        vecs = np.concatenate((np.zeros((1, embedding_dims), np.float32), np.random.standard_normal((len(vocab), embedding_dims))))
        word2index = {}
        for i, tok in enumerate(vocab):
            vecs[i + 1] = w2v_dict[tok]
            word2index[tok] = i + 1

        pickle.dump({'word2index' : word2index, 'vecs' : vecs}, open(cachefn, 'wb'))

    return word2index, vecs

class DatasetLoader:
    """ Dataset loader class that loads feature matrices from given paths and
        create shuffled batch for training, unshuffled batch for evaluation.
    """
    def __init__(self, args, split):
        im_feats = np.load(os.path.join('data', args.dataset, split + '.npy'))
        with open(os.path.join('data', args.dataset, split + '.txt'), 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]

        assert len(im_feats) == len(image_ids)
        self.image2index = dict(zip(image_ids, range(len(image_ids))))
        self.split = split
        self.im_feat_shape = (im_feats.shape[0], im_feats.shape[-1])
        self.im_feats = im_feats
        self.languages = args.languages
        self.sent_feats = {}
        self.num_sentences = {}
        self.sent2im = {}
        self.im2sent = {}
        self.vecs = {}
        self.vocab = {}
        self.max_length = {}
        max_sentences = 0
        for language in args.languages:
            tokens, sent2im, im2sent, vocab, max_length = self.get_tokens(args, language)
            self.vocab[language], self.vecs[language] = get_embeddings(args, language, vocab)
            language_features = get_sentence(self.vocab[language], tokens, max_length)
            self.max_length[language] = max_length
            self.sent_feats[language] = language_features
            num_sentences = len(sent2im)
            self.num_sentences[language] = num_sentences
            if num_sentences > max_sentences:
                self.sent2im = sent2im
                self.max_language = language
                max_sentences = num_sentences

            self.im2sent[language] = im2sent

        self.sent_inds = range(max_sentences)
        if split != 'train':
            self.test_labels = {}
            for language, im2sent in self.im2sent.iteritems():
                labels = np.zeros((self.num_sentences[language], len(self.image2index)), np.bool)
                for image_index, sentences in im2sent.iteritems():
                    labels[sentences, image_index] = True

                self.test_labels[language] = labels

    @abstractmethod
    def get_tokens(self, args, language):
        pass

    def shuffle_inds(self):
        '''
        shuffle the indices in training (run this once per epoch)
        nop for testing and validation
        '''
        np.random.shuffle(self.sent_inds)

    def sample_items(self, sample_inds, sample_size):
        '''
        for each index, return the  relevant image and sentence features
        sample_inds: a list of sent indices
        sample_size: number of neighbor sentences to sample per index.
        '''
        im_ids = [self.sent2im[i] for i in sample_inds]
        im_feats = self.im_feats[im_ids]
        sent_feats = {}
        for language in self.languages:
            feats = []
            for im, sent in zip(im_ids, sample_inds):
                im_sent = list(self.im2sent[language][im])
                if language == self.max_language:
                    if len(im_sent) < sample_size:
                        for _ in range(sample_size - len(im_sent)):
                            im_sent.append(np.random.choice(im_sent))

                        sample_index = im_sent
                    else:
                        im_sent.remove(sent)
                        sample_index = np.random.choice(im_sent, sample_size - 1, replace=False)
                        sample_index = np.append(sample_index, sent)
                else:
                    sample_index = np.random.choice(im_sent, sample_size, replace=len(im_sent) < sample_size)

                feats.append(self.sent_feats[language][sample_index])

            sent_feats[language] = np.concatenate(feats, axis=0)

        return (im_feats, sent_feats)

    def get_batch(self, batch_index, batch_size, sample_size):
        start_ind = batch_index * batch_size
        end_ind = start_ind + batch_size
        sample_inds = self.sent_inds[start_ind : end_ind]
        (im_feats, sent_feats) = self.sample_items(sample_inds, sample_size)

        # Each row of the labels is the label for one sentence,
        # with corresponding image index sent to True.
        labels = np.repeat(np.eye(batch_size, dtype=bool), sample_size, axis=0)
        return (im_feats, sent_feats, labels)

    


