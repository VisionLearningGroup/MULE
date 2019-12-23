from __future__ import division
from __future__ import print_function

import argparse
import sys
import json
import os

import numpy as np
import tensorflow as tf

from multi30k import Multi30KLoader
from coco import COCOLoader
from retrieval_model import MULE, recall_k

def get_image_embedding(args, restore_path, data_loader):
    im_feat_plh = tf.placeholder(tf.float32, shape=data_loader.im_feat_shape)
    train_phase_plh = tf.placeholder(tf.bool)
    model = MULE(args, data_loader.vecs)
    features = model.setup_img_model(im_feat_plh, train_phase_plh)
    saver = tf.train.Saver(save_relative_paths=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Getting image embedding')

        # Restore latest checkpoint or the given MetaGraph.
        if restore_path.endswith('.meta'):
            ckpt_path = restore_path.replace('.meta', '')
        else:
            ckpt_path = tf.train.latest_checkpoint(restore_path)

        print('Restoring checkpoint', ckpt_path)
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        print('Done')
        feed_dict = {train_phase_plh : False,
                     im_feat_plh : data_loader.im_feats}

        image_embedding = sess.run(features, feed_dict = feed_dict)

    tf.reset_default_graph()
    return image_embedding

def get_sentence_embedding(args, restore_path, data_loader):
    print('Getting sentence embedding')
    sent_feats_plh = {}
    for language in args.languages:
        sent_feats_plh[language] = tf.placeholder(tf.int32, shape=data_loader.sent_feats[language].shape)

    train_phase_plh = tf.placeholder(tf.bool)
    model = MULE(args, data_loader.vecs)
    features, _, _ = model.setup_sent_model(sent_feats_plh, train_phase_plh)
    saver = tf.train.Saver(save_relative_paths=True)
    with tf.Session() as sess:
        # Restore latest checkpoint or the given MetaGraph.
        if restore_path.endswith('.meta'):
            ckpt_path = restore_path.replace('.meta', '')
        else:
            ckpt_path = tf.train.latest_checkpoint(restore_path)

        print('Restoring checkpoint', ckpt_path)
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        print('Done')
        feed_dict = {train_phase_plh : False}
        for language, placeholder in sent_feats_plh.iteritems():
            feed_dict[placeholder] = data_loader.sent_feats[language]

        sentence_embeddings = sess.run(features, feed_dict = feed_dict)

    tf.reset_default_graph()
    return sentence_embeddings

def test_epoch(args, restore_path, data_loader):
    image_features = get_image_embedding(args, restore_path, data_loader)
    sentence_features = get_sentence_embedding(args, restore_path, data_loader)
    all_recalls = np.zeros(6)
    for language, sentences in sentence_features.iteritems():
        im_feat_plh = tf.placeholder(tf.float32, shape=image_features.shape)
        sent_feat_plh = tf.placeholder(tf.float32, shape=sentences.shape)
        label_plh = tf.placeholder(tf.bool, shape=[len(sentences), len(image_features)])
        performance, _ = recall_k(im_feat_plh, sent_feat_plh, label_plh, ks=tf.convert_to_tensor([1,5,10]))
        with tf.Session() as sess:
            feed_dict = {im_feat_plh : image_features,
                         sent_feat_plh : sentences,
                         label_plh : data_loader.test_labels[language]}

            recalls = sess.run(performance, feed_dict = feed_dict)
            recalls = np.round(recalls * 100, 1)
            print(language, 'im2sent:', ' '.join(map(str, recalls[:3])),
                  'sent2im:', ' '.join(map(str, recalls[3:])),
                  'mR: ', round(np.mean(recalls), 1))

            all_recalls += recalls

        tf.reset_default_graph()

    average_recall = np.mean(all_recalls) / float(len(sentence_features))
    return average_recall

def main(_):
    json_file = os.path.join(os.path.split(args.restore_path)[0], 'results.json')
    with open(json_file, 'r') as f:
        config = json.load(f)

    args.separate_lang_branch = config['separate_lang_branch']
    args.max_sentence_length = config['max_sentence_length']

    # Load data.
    if args.dataset == 'multi30k':
        data_loader = Multi30KLoader(args, args.split)
    else:
        data_loader = COCOLoader(args, args.split)

    if not os.path.exists(args.restore_path):
        epoch = config['val_best_epoch']
        print('Epoch not found, loading best val epoch %i' % epoch)
        args.restore_path = os.path.join(os.path.split(args.restore_path)[0], 'two_branch-ckpt-%i.meta' % epoch)

    test_epoch(args, args.restore_path, data_loader)

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--split', type=str, default='train', help='Dataset split to train with.')
    parser.add_argument('--dataset', type=str, help='Dataset to train.')
    parser.add_argument('--languages', type=str, help='List of languages to train.')
    parser.add_argument('--restore_path', type=str,
                        help='Directory for restoring the newest checkpoint or\
                              path to a restoring checkpoint MetaGraph file.')

    global args
    args, unparsed = parser.parse_known_args()
    args.languages = args.languages.split(',')
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
