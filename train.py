from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import json

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from multi30k import Multi30KLoader
from coco import COCOLoader
from retrieval_model import MULE
from test import test_epoch

def process_epoch(epoch, data_loader, restore_path, vecs, pretrain_univeral_embedding):
    im_feat_dim = data_loader.im_feat_shape[1]
    steps_per_epoch = len(data_loader.sent_inds) // args.batch_size

    # Setup placeholders for input variables.
    im_feat_plh = tf.placeholder(tf.float32, shape=[args.batch_size, im_feat_dim])
    sent_feat_plh = {}
    # this list controls the languages a model actually learns
    for language in args.languages:
        token_length = data_loader.max_length[language]
        sent_feat_plh[language] = tf.placeholder(tf.int32, shape=[args.batch_size * args.sample_size, token_length])

    label_plh = tf.placeholder(tf.bool, shape=[args.batch_size * args.sample_size, args.batch_size])
    train_phase_plh = tf.placeholder(tf.bool)

    # Setup training operation.
    model = MULE(args, vecs)
    loss = model.setup_train_model(im_feat_plh, sent_feat_plh, train_phase_plh, label_plh)

    # Setup optimizer.
    global_step = tf.Variable(epoch * steps_per_epoch, trainable=False)
    learning_rate = tf.train.exponential_decay(args.lr, global_step,
                                               steps_per_epoch, 0.794, staircase=True)

    optim = tf.train.AdamOptimizer(args.lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optim.minimize(loss, global_step=global_step)

    # Setup model saver.
    saver = tf.train.Saver(save_relative_paths=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if pretrain_univeral_embedding:
            print('restoring pretrain checkpoint', restore_path)
            mule_embedding_names = []
            for language in args.languages:
                mule_embedding_names.append('mule_%s/weights:0' % language)
                mule_embedding_names.append('mule_%s/biases:0' % language)
            
            variables = tf.contrib.slim.get_variables_to_restore()
            variables_to_restore = [v for v in variables if v.name in mule_embedding_names]
            pretrain_restore = tf.train.Saver(variables_to_restore, save_relative_paths=True)
            pretrain_restore.restore(sess, restore_path.replace('.meta', ''))
            print('done')
        elif restore_path:
            print('restoring checkpoint', restore_path)
            saver.restore(sess, restore_path.replace('.meta', ''))
            print('done')

        data_loader.shuffle_inds()
        for i in range(steps_per_epoch):
            im_feats, sent_feats, labels = data_loader.get_batch(i, args.batch_size, args.sample_size)
            p = float(epoch * steps_per_epoch + i) / (steps_per_epoch * args.max_num_epoch)
            dm_lr = (2. / (1. + np.exp(-10. * p)) - 1) * args.domain_adapt
            feed_dict = {
                    im_feat_plh : im_feats,
                    label_plh : labels,
                    model.dm_lr : dm_lr,
                    train_phase_plh : True,
            }

            for language, lang_plh in sent_feat_plh.iteritems():
                feed_dict[lang_plh] = sent_feats[language]

            [_, loss_val] = sess.run([train_step, loss], feed_dict = feed_dict)
            if i % 100 == 0:
                print('Epoch: [%d/%d] Step: [%d/%d] Loss: %f' % (epoch + 1, args.max_num_epoch, i + 1, steps_per_epoch, loss_val))

        print('Saving checkpoint')
        restore_path = os.path.join(args.save_dir, 'two_branch-ckpt')
        saver.save(sess, restore_path, global_step = epoch + 1)
        
        embeddings = {}
        for language in args.languages:
            embeddings['word_embeddings_%s:0' % language] = language

        variables = tf.GraphKeys.GLOBAL_VARIABLES
        for v in tf.get_collection(variables):
            if v.name in embeddings:
                vecs[embeddings[v.name]] = v.eval()

    print(restore_path)
    tf.reset_default_graph()
    return vecs, restore_path + '-%i' % (epoch + 1)

def train_language_universal_embedding(data_loader, vecs):
    steps_per_epoch = len(data_loader.sent_inds) // args.batch_size
    sent_feat_plh = {}
    # this list controls the languages a model actually learns
    for language in args.languages:
        token_length = data_loader.max_length[language]
        sent_feat_plh[language] = tf.placeholder(tf.int32, shape=[args.batch_size * args.sample_size, token_length])

    label_plh = tf.placeholder(tf.bool, shape=[args.batch_size * args.sample_size, args.batch_size])
    model = MULE(args, vecs)
    loss = model.universal_embedding_train(sent_feat_plh, label_plh)

    # Setup optimizer.
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(args.lr, global_step,
                                               steps_per_epoch, 0.794, staircase=True)
    optim = tf.train.AdamOptimizer(args.lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optim.minimize(loss, global_step=global_step)

    # Setup model saver.
    saver = tf.train.Saver(save_relative_paths=True)
    num_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in tqdm(range(num_epochs),desc='pretraining embedding', total=num_epochs):
            data_loader.shuffle_inds()
            for i in range(steps_per_epoch):
                im_feats, sent_feats, labels = data_loader.get_batch(i, args.batch_size, args.sample_size)
                feed_dict = {label_plh : labels}
                for language, lang_plh in sent_feat_plh.iteritems():
                    feed_dict[lang_plh] = sent_feats[language]

                [_, loss_val] = sess.run([train_step, loss], feed_dict = feed_dict)

        print('Saving checkpoint')
        restore_path = os.path.join(args.save_dir, 'two_branch-ckpt')
        saver.save(sess, restore_path, global_step = 0)

    tf.reset_default_graph()
    return restore_path + '-0'

def main(_):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load data.
    if args.dataset == 'multi30k':
        train_data_loader = Multi30KLoader(args, 'train')
        val_data_loader = Multi30KLoader(args, 'val')
    else:
        train_data_loader = COCOLoader(args, 'train')
        val_data_loader = COCOLoader(args, 'val')

    vecs = train_data_loader.vecs
    restore_path = args.restore_path
    best_perf = 0.
    best_epoch = 0
    for epoch in range(args.max_num_epoch):
        pretrain_univeral_embedding = args.univ_pretrain and epoch == 0 and not restore_path
        if epoch % 5 == 0 or epoch == args.max_num_epoch:
            if pretrain_univeral_embedding:
                restore_path = train_language_universal_embedding(train_data_loader, vecs)

        vecs, restore_path = process_epoch(epoch, train_data_loader, restore_path, vecs, pretrain_univeral_embedding)
        perf = test_epoch(args, restore_path + '.meta', val_data_loader)
        if perf > best_perf:
            best_perf = perf
            best_epoch = epoch + 1

    args.val_best_perf = best_perf
    args.val_best_epoch = best_epoch
    json_outfile = os.path.join(args.save_dir, 'results.json')
    print('best epoch at %i with val score %.1f' % (best_epoch, best_perf))
    with open(json_outfile, 'w') as outfile:
        json.dump(vars(args), outfile)

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--save_dir', type=str, help='Directory for saving checkpoints.')
    parser.add_argument('--restore_path', type=str, help='Path to the restoring checkpoint MetaGraph file.')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to train with.')
    parser.add_argument('--dataset', type=str, help='Dataset to train.')
    parser.add_argument('--languages', type=str, help='List of languages to train.')
    # Training parameters.
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=450, help='Batch size for training.')
    parser.add_argument('--sample_size', type=int, default=2, help='Number of positive pair to sample.')
    parser.add_argument('--max_sentence_length', type=int, default=40, help='Maximum number of words allowed per sentence.')
    parser.add_argument('--max_num_epoch', type=int, default=20, help='Max number of epochs to train.')
    parser.add_argument('--num_neg_sample', type=int, default=10, help='Number of negative example to sample.')
    parser.add_argument('--margin', type=float, default=0.05, help='Margin.')
    parser.add_argument('--embed_reg', type=float, default=5e-7, help='Word embedding regularization weight.')
    parser.add_argument('--uni_align', type=float, default=1., help='MULE embedding neighborhood constraint weight.')
    parser.add_argument('--embed_align', type=float, default=1., help='Cross-lang neighborhood constraint weight.')
    parser.add_argument('--domain_adapt', type=float, default=1e-6, help='Language classifier loss weight.')
    parser.add_argument('--im_loss_factor', type=float, default=1.5,
                        help='Factor multiplied with image loss. Set to 0 for single direction.')
    parser.add_argument('--sent_only_loss_factor', type=float, default=0.05,
                        help='Factor multiplied with mono-lang sent only loss. Set to 0 for no neighbor constraint.')
    parser.add_argument('--univ_pretrain', action='store_true', default=False, help='Pretrain the university embedding')
    parser.add_argument('--separate_lang_branch', action='store_true', default=False, help='When true, a different language branch is used for each language')

    global args
    args, unparsed = parser.parse_known_args()
    args.languages = args.languages.split(',')
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
