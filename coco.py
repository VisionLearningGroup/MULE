from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
from data_loader import DatasetLoader

class COCOLoader(DatasetLoader):
    """ Dataset loader class that loads feature matrices from given paths and
        create shuffled batch for training, unshuffled batch for evaluation.
    """
    def tokenize(self, language, token_filename, image_filename, tokens, vocab):
        with open(token_filename, 'r') as f:
            all_sentences = json.load(f)

        image_list = [im.strip() for im in open(image_filename, 'r').readlines()]
        assert len(all_sentences) == len(image_list)
        max_length = 0
        for im, sentences in zip(image_list, all_sentences):
            i = self.image2index[im]
            for sentence in sentences:
                if language == 'en':
                    sentence = sentence.lower().split()
                else:
                    sentence = sentence.encode('utf8').split()

                vocab.update(sentence)
                max_length = max(len(sentence), max_length)
                tokens[i].append(sentence)

        return max_length

    def get_tokens(self, args, language):
        token_filename = os.path.join('data', args.dataset, 'tokenized', '%s_%s_caption_list.json' % (self.split, language))
        if self.split != 'train':
            image_filename = os.path.join('data', args.dataset, self.split + '.txt')
        elif language == 'en':
            # contains images which there are no human-generated sentences for other languages
            image_filename = os.path.join('data', args.dataset, 'tokenized', '%s_%s_coco.txt' % (self.split, language))
        else:
            image_filename = os.path.join('data', args.dataset, 'tokenized', '%s_en_%s_coco.txt' % (self.split, language))

        tokens = [[] for _ in range(len(self.image2index))]
        vocab = set()
        max_length = self.tokenize(language, token_filename, image_filename, tokens, vocab)
        if self.split == 'train':
            if language == 'en':
                # add images that have human-generated japanese captions
                token_filename = os.path.join('data', args.dataset, 'tokenized', 'train_en_jp_caption_list.json')
                image_filename = os.path.join('data', args.dataset, 'tokenized', 'train_en_jp_coco.txt')
                max_length = max(max_length, self.tokenize(language, token_filename, image_filename, tokens, vocab))

                # add images that have human-generated chinese captions
                token_filename = os.path.join('data', args.dataset, 'tokenized', 'train_en_cn_caption_list.json')
                image_filename = os.path.join('data', args.dataset, 'tokenized', 'train_en_cn_coco.txt')
                max_length = max(max_length, self.tokenize(language, token_filename, image_filename, tokens, vocab))

                # add translations to english from other languages
                for this_lang in ['cn', 'jp']:
                    token_filename = os.path.join('data', args.dataset, 'tokenized', 'train_%s_to_en_caption_list.json' % this_lang)
                    image_filename = os.path.join('data', args.dataset, 'tokenized', 'train_en_%s_coco.txt' % this_lang)
                    max_length = max(max_length, self.tokenize(language, token_filename, image_filename, tokens, vocab))                        
                    
            else:
                # add translations from english
                token_filename = os.path.join('data', args.dataset, 'tokenized', 'train_%s_augment_caption_list.json' % language)
                image_filename = os.path.join('data', args.dataset, 'tokenized', 'train_en_%s_augment_coco.txt' % language)
                max_length = max(max_length, self.tokenize(language, token_filename, image_filename, tokens, vocab))

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
