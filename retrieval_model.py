import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected
from flip_gradient import flip_gradient

def pdist(x1, x2):
    """
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = tf.reshape(tf.reduce_sum(x1*x1, axis=1), [-1, 1])
    x2_square = tf.reshape(tf.reduce_sum(x2*x2, axis=1), [1, -1])
    return tf.sqrt(x1_square - 2 * tf.matmul(x1, tf.transpose(x2)) + x2_square + 1e-4)

def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res

def recall_k(im_embeds, sent_embeds, im_labels, ks=None):
    """
        Compute recall at given ks.
    """
    sent_im_dist = pdist(sent_embeds, im_embeds)
    def retrieval_recall(dist, labels, k):
        # Use negative distance to find the index of
        # the smallest k elements in each row.
        pred = tf.nn.top_k(-dist, k=k)[1]
        # Create a boolean mask for each column (k value) in pred,
        # s.t. mask[i][j] is 1 iff pred[i][k] = j.
        pred_k_mask = lambda topk_idx: tf.one_hot(topk_idx, labels.shape[1],
                            on_value=True, off_value=False, dtype=tf.bool)
        # Create a boolean mask for the predicted indicies
        # by taking logical or of boolean masks for each column,
        # s.t. mask[i][j] is 1 iff j is in pred[i].
        pred_mask = tf.reduce_any(tf.map_fn(
                pred_k_mask, tf.transpose(pred), dtype=tf.bool), axis=0)
        # Entry (i, j) is matched iff pred_mask[i][j] and labels[i][j] are 1.
        matched = tf.cast(tf.logical_and(pred_mask, labels), dtype=tf.float32)
        return tf.reduce_mean(tf.reduce_max(matched, axis=1))
    return tf.concat(
        [tf.map_fn(lambda k: retrieval_recall(tf.transpose(sent_im_dist), tf.transpose(im_labels), k),
                   ks, dtype=tf.float32),
         tf.map_fn(lambda k: retrieval_recall(sent_im_dist, im_labels, k),
                   ks, dtype=tf.float32)],
        axis=0), sent_im_dist

def add_fc(inputs, outdim, train_phase, scope_in):
    fc =  fully_connected(inputs, outdim, activation_fn=None, scope=scope_in + '/fc')
    fc_bnorm = tf.layers.batch_normalization(fc, momentum=0.1, epsilon=1e-5,
                         training=train_phase, name=scope_in + '/bnorm')
    fc_relu = tf.nn.relu(fc_bnorm, name=scope_in + '/relu')
    fc_out = tf.layers.dropout(fc_relu, seed=0, training=train_phase, name=scope_in + '/dropout')
    return fc_out

def universal_embedding_layer(embedded_word_ids, tokens, embed_dim, suffix, trainable=True):
    universal_embedding = fully_connected(embedded_word_ids, embed_dim, activation_fn = None,
                                          weights_regularizer = tf.contrib.layers.l2_regularizer(0.005),
                                          trainable=trainable,
                                          scope = 'mule_' + suffix)

    num_words = tf.reduce_sum(tf.to_float(tokens > 0), 1, keep_dims=True) + 1e-10
    avg_words = tf.nn.l2_normalize(tf.reduce_sum(universal_embedding, 1) / num_words, 1)
    return universal_embedding, avg_words

def setup_lstm(encoder_cell, embedded_word_ids, tokens, source_sequence_length, fc_dim, embed_dim, reuse, suffix):
    universal_embedding, avg_words = universal_embedding_layer(embedded_word_ids, tokens, embed_dim, suffix)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        encoder_cell, universal_embedding, dtype=tf.float32,
        sequence_length=source_sequence_length, scope='rnn')

    final_outputs = extract_axis_1(encoder_outputs, source_sequence_length-1)
    outputs = fully_connected(final_outputs, embed_dim, activation_fn = None,
                              weights_regularizer = tf.contrib.layers.l2_regularizer(0.005),
                              scope = 'phrase_encoder', reuse=reuse)

    sent_embed = tf.nn.l2_normalize(outputs, 1, epsilon=1e-10)
    return sent_embed, avg_words

class MULE():
    def __init__(self, args, vecs):
        self.fc_dim = 2048
        self.embed_dim = 512
        self.hidden_dim = 1024
        self.embeddings = vecs
        self.is_train = args.split == 'train'
        self.args = args
        self.dm_lr = tf.placeholder(tf.float32, [])
            
    def embedding_loss(self, im_embeds, sent_embeds, im_labels):
        """
            im_embeds: (b, 512) image embedding tensors
            sent_embeds: (sample_size * b, 512) sentence embedding tensors
                where the order of sentence corresponds to the order of images and
                setnteces for the same image are next to each other
            im_labels: (sample_size * b, b) boolean tensor, where (i, j) entry is
                True if and only if sentence[i], image[j] is a positive pair
        """
        # compute embedding loss
        sent_im_ratio = self.args.sample_size
        num_img = self.args.batch_size
        num_sent = num_img * sent_im_ratio
        
        sent_im_dist = pdist(sent_embeds, im_embeds)
        # image loss: sentence, positive image, and negative image
        pos_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, im_labels), [num_sent, 1])
        neg_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, ~im_labels), [num_sent, -1])
        im_loss = tf.clip_by_value(self.args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
        im_loss = tf.reduce_mean(tf.nn.top_k(im_loss, k=self.args.num_neg_sample)[0])
        # sentence loss: image, positive sentence, and negative sentence
        neg_pair_dist = tf.reshape(tf.boolean_mask(tf.transpose(sent_im_dist), ~tf.transpose(im_labels)), [num_img, -1])
        neg_pair_dist = tf.reshape(tf.tile(neg_pair_dist, [1, sent_im_ratio]), [num_sent, -1])
        sent_loss = tf.clip_by_value(self.args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
        sent_loss = tf.reduce_mean(tf.nn.top_k(sent_loss, k=self.args.num_neg_sample)[0])
        # sentence only loss (neighborhood-preserving constraints)
        sent_sent_dist = pdist(sent_embeds, sent_embeds)
        sent_sent_mask = tf.reshape(tf.tile(tf.transpose(im_labels), [1, sent_im_ratio]), [num_sent, num_sent])
        pos_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, sent_sent_mask), [-1, sent_im_ratio])
        pos_pair_dist = tf.reduce_max(pos_pair_dist, axis=1, keep_dims=True)
        neg_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, ~sent_sent_mask), [num_sent, -1])
        sent_only_loss = tf.clip_by_value(self.args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
        sent_only_loss = tf.reduce_mean(tf.nn.top_k(sent_only_loss, k=self.args.num_neg_sample)[0])
        
        loss = im_loss * self.args.im_loss_factor + sent_loss + sent_only_loss * self.args.sent_only_loss_factor
        return loss

    def setup_img_model(self, im_feats, train_phase):
        im_fc1 = add_fc(im_feats, self.fc_dim, train_phase, 'im_embed_1')
        im_fc2 = fully_connected(im_fc1, self.embed_dim, activation_fn=None,
                                 scope = 'im_embed_2')
        i_embed = tf.nn.l2_normalize(im_fc2, 1, epsilon=1e-10)
        return i_embed

    def setup_lstm(self, encoder_cell, embedded_word_ids, tokens, source_sequence_length, reuse, suffix):
        universal_embedding, avg_words = universal_embedding_layer(embedded_word_ids, tokens, self.embed_dim, suffix)
        if self.args.separate_lang_branch:
            reuse = None
            suffix = '_' + suffix
        else:
            suffix = ''

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, universal_embedding, dtype=tf.float32,
            sequence_length=source_sequence_length, scope='rnn' + suffix)

        final_outputs = extract_axis_1(encoder_outputs, source_sequence_length-1)
        outputs = fully_connected(final_outputs, self.embed_dim, activation_fn = None,
                                  weights_regularizer = tf.contrib.layers.l2_regularizer(0.005),
                                  scope = 'phrase_encoder' + suffix, reuse=reuse)

        sent_embed = tf.nn.l2_normalize(outputs, 1, epsilon=1e-10)
        return sent_embed, avg_words

    def setup_sent_model(self, all_tokens, train_phase):
        reuse = None
        sent_embed, embed_l2reg, avg_words = {}, [], {}
        for language in self.args.languages:
            tokens = all_tokens[language]
            embedding_init = self.embeddings[language]
            word_embeddings = tf.get_variable('word_embeddings_' + language, embedding_init.shape, initializer=tf.constant_initializer(embedding_init))
            embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, tokens)
            embed_l2reg.append(tf.nn.l2_loss(word_embeddings - embedding_init))
            
            reuse_gru = not self.args.separate_lang_branch and reuse
            encoder_cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim, reuse=reuse_gru)

            source_sequence_length = tf.reduce_sum(tf.cast(tokens > 0, tf.int32), 1)
            embed, avg = self.setup_lstm(encoder_cell, embedded_word_ids, tokens, source_sequence_length, reuse, language)
            sent_embed[language] = embed
            avg_words[language] = avg
            reuse = True

        return sent_embed, embed_l2reg, avg_words

    def sentence_loss(self, sent_embeds, im_embeds, im_labels):
        """
            im_embeds: (b, 512) image embedding tensors
            sent_embeds: (sample_size * b, 512) sentence embedding tensors
                where the order of sentence corresponds to the order of images and
                setnteces for the same image are next to each other
            im_labels: (sample_size * b, b) boolean tensor, where (i, j) entry is
                True if and only if sentence[i], image[j] is a positive pair
        """
        # compute embedding loss
        sent_im_ratio = self.args.sample_size
        num_img = self.args.batch_size
        num_sent = num_img * sent_im_ratio
        sent_im_dist = pdist(sent_embeds, im_embeds)

        # sentence only loss (neighborhood-preserving constraints)
        sent_sent_mask = tf.reshape(tf.tile(tf.transpose(im_labels), [1, sent_im_ratio]), [num_sent, num_sent])
        pos_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, sent_sent_mask), [-1, sent_im_ratio])
        pos_pair_dist = tf.reduce_max(pos_pair_dist, axis=1, keep_dims=True)
        neg_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, ~sent_sent_mask), [num_sent, -1])
        sent_only_loss = tf.clip_by_value(self.args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
        im_loss = tf.reduce_mean(tf.nn.top_k(sent_only_loss, k=self.args.num_neg_sample)[0])
        return im_loss

    def domain_classifier_layer(self, sent_feats, reuse = None, trainable = False, params = None):
        if params is None:
            weight_init = tf.contrib.layers.xavier_initializer()
            bias_init = tf.zeros_initializer()
        else:
            weight_init = tf.constant_initializer(params['weights'])
            bias_init = tf.constant_initializer(params['biases'])

        outputs = fully_connected(sent_feats, len(self.args.languages), activation_fn = None,
                                  weights_initializer=weight_init,
                                  biases_initializer=bias_init,
                                  trainable=trainable,
                                  weights_regularizer = tf.contrib.layers.l2_regularizer(0.005),
                                  scope = 'domain_language_classifier', reuse=reuse)
        return outputs

    def universal_embedding_train(self, all_tokens, im_labels):
        all_sent = {}
        num_lang = len(all_tokens)
        num_items = self.args.batch_size * self.args.sample_size
        for language in self.args.languages:
            tokens = all_tokens[language]
            embedding_init = self.embeddings[language]
            word_embeddings = tf.get_variable('word_embeddings_' + language, embedding_init.shape, initializer=tf.constant_initializer(embedding_init), trainable=False)
            embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, tokens)
            _, avg_words = universal_embedding_layer(embedded_word_ids, tokens, self.embed_dim, language)
            all_sent[language] = avg_words

        universal_align_loss= []
        for i in range(len(all_sent)-1):
            lang1 = self.args.languages[i]
            for j in range(i+1, len(all_sent)):
                lang2 = self.args.languages[j]
                universal_align_loss.append(self.sentence_loss(all_sent[lang1], all_sent[lang2], im_labels))

        universal_align_loss = tf.reduce_mean(universal_align_loss)
        return universal_align_loss

    def domain_loss(self, sent_feats, true_label, reuse = None):
        feat = flip_gradient(sent_feats, self.dm_lr)
        num_lang = len(self.args.languages)
        outputs = self.domain_classifier_layer(feat, reuse, trainable=True)
        num_items = self.args.batch_size * self.args.sample_size
        indices = tf.expand_dims(tf.range(true_label, num_items * num_lang, num_lang), 1)
        labels = tf.ones(num_items, tf.float32)
        labels = tf.reshape(tf.scatter_nd(indices, labels, [num_items * num_lang]), [-1, num_lang])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs))
        return loss

    def setup_train_language_model(self, sent_feats, train_phase, im_labels):
        sent_embed, embed_reg, avg_words = self.setup_sent_model(sent_feats, train_phase)
        embed_reg = tf.reduce_mean(embed_reg)
        dm_loss = universal_align_loss = embedding_align_loss = 0.
        if len(self.args.languages) > 1:
            if self.args.domain_adapt > 0:
                dm_loss = []
                reuse = None
                for i, language in enumerate(self.args.languages):
                    dm_loss.append(self.domain_loss(avg_words[language], i, reuse))
                    reuse = True

                dm_loss = tf.reduce_mean(dm_loss)

            universal_align_loss, embedding_align_loss = [], []
            for i in range(len(sent_embed)-1):
                lang1 = self.args.languages[i]
                for j in range(i+1, len(sent_embed)):
                    lang2 = self.args.languages[j]
                    universal_align_loss.append(self.sentence_loss(sent_embed[lang1], sent_embed[lang2], im_labels))
                    embedding_align_loss.append(self.sentence_loss(avg_words[lang1], avg_words[lang2], im_labels))

            universal_align_loss, embedding_align_loss = tf.reduce_mean(universal_align_loss), tf.reduce_mean(embedding_align_loss)

        lang_loss = universal_align_loss * self.args.uni_align + embedding_align_loss * self.args.embed_align + dm_loss * self.args.domain_adapt + embed_reg * self.args.embed_reg
        return lang_loss, sent_embed

    def setup_train_model(self, im_feats, sent_feats, train_phase, im_labels):
        # im_feats b x image_feature_dim
        # sent_feats 5b x sent_feature_dim
        # train_phase bool (Should be True.)
        # im_labels 5b x b
        i_embed = self.setup_img_model(im_feats, train_phase)
        lang_loss, sent_embed = self.setup_train_language_model(sent_feats, train_phase, im_labels)
        loss = []
        for language in self.args.languages:
            loss.append(self.embedding_loss(i_embed, sent_embed[language], im_labels))

        total_loss = tf.reduce_mean(loss) + lang_loss
        return total_loss




