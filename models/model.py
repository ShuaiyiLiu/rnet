# TODO: add bias
import math
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from .rnn_cells import mat_weight_mul, GatedAttentionGRUCell, attention_pooling


class RNet:
    @staticmethod
    def dropout_wrapped_grucell(hidden_size, in_keep_prob):
        cell = tf.contrib.rnn.GRUCell(hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    @staticmethod
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def cross_entropy(labels, predict):
        predict = tf.clip_by_value(predict, 1e-8, 1.0)
        return -tf.reduce_sum(labels * tf.log(predict), 1)

    def __init__(self, options):
        self.options = options
        h = options['h_size']


    def build_model(self, it):
        options = self.options
        # placeholders
        eP = it['eP']
        eQ = it['eQ']
        asi = it['asi']
        aei = it['aei']

        p_max_length = options['p_length']
        q_max_length = options['q_length']
        h_size = options['h_size']
        in_keep_prob = options['in_keep_prob']

        num_of_layers = 3 # TODO: add to options

        # embeddings concatenation
        # TODO: add character level embeddings
        eQcQ = eQ
        ePcP = eP
        qlen = self.length(eQ)
        plen = self.length(eP)
        p_mask = tf.sequence_mask(plen, p_max_length, dtype=tf.float32)
        q_mask = tf.sequence_mask(qlen, q_max_length, dtype=tf.float32)

        # Question and Passage Encoder
        with tf.variable_scope('encoder') as scope:
            gru_cells_fw = tf.nn.rnn_cell.MultiRNNCell([self.dropout_wrapped_grucell(h_size, in_keep_prob)
                                                        for _ in range(num_of_layers)])
            gru_cells_bw = tf.nn.rnn_cell.MultiRNNCell([self.dropout_wrapped_grucell(h_size, in_keep_prob)
                                                        for _ in range(num_of_layers)])

            uQ_2, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cells_fw, gru_cells_bw, eQcQ, dtype=tf.float32, scope='context_encoding')
            scope.reuse_variables()

            uP_2, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cells_fw, gru_cells_bw, ePcP, dtype=tf.float32, scope='context_encoding')
            uQ = tf.concat(uQ_2, 2)
            uP = tf.concat(uP_2, 2)
            uQ = tf.nn.dropout(uQ, in_keep_prob)
            uP = tf.nn.dropout(uP, in_keep_prob)

        # Question and passage matching
        with tf.variable_scope('attention_matching'):
            attn_cells_fw = GatedAttentionGRUCell(h_size, uQ, 2 * h_size, 2 * h_size, True)
            attn_cells_bw = GatedAttentionGRUCell(h_size, uQ, 2 * h_size, 2 * h_size, True)
            vP_2, _ = tf.nn.bidirectional_dynamic_rnn(attn_cells_fw, attn_cells_bw, dtype=tf.float32, inputs=uP)
            vP = tf.concat(vP_2, 2)
            vP = tf.nn.dropout(vP, in_keep_prob)
            print('Shape of vP: {}'.format(vP.get_shape()))

        # self matching layer
        with tf.variable_scope('self_matching'):
            attn_sm_cells_fw = GatedAttentionGRUCell(h_size, vP, 2 * h_size, 2 * h_size, False)
            attn_sm_cells_bw = GatedAttentionGRUCell(h_size, vP, 2 * h_size, 2 * h_size, False)
            hP_2, _ = tf.nn.bidirectional_dynamic_rnn(attn_sm_cells_fw,
                                                      attn_sm_cells_bw,
                                                      inputs=vP,
                                                      dtype=tf.float32)
            hP = tf.concat(hP_2, 2)
            hP = tf.nn.dropout(hP, in_keep_prob)
            print('Shape of hP: {}'.format(hP.get_shape()))

        # output layer
        with tf.variable_scope('encode_again'):
            # quote in section 4.2: After the original self-matching layer of the passage,
            # we utilize bi-directional GRU to deeply integrate the matching
            # results before feeding them into answer pointer layer.
            gru_cells_fw2 = tf.contrib.rnn.GRUCell(h_size)
            gru_cells_bw2 = tf.contrib.rnn.GRUCell(h_size)
            gP_2, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cells_fw2, gru_cells_bw2, hP, dtype=tf.float32, scope='deeply_integration')
            hP = tf.concat(gP_2, 2)

        # question pooling
        with tf.variable_scope('question_pooling'):
            WuQ = tf.get_variable(
                'WuQ', shape=[h_size * 2, h_size], dtype=tf.float32, initializer=xavier_initializer())
            WuQ_uQ = mat_weight_mul(uQ, WuQ)  # batch_size x q_length x H
            WvQVrQ = tf.get_variable(
                'WvQVrQ', shape=[1, h_size], dtype=tf.float32, initializer=xavier_initializer())
            v0 = tf.get_variable('v0', shape=[h_size], dtype=tf.float32, initializer=xavier_initializer())
            rQ = attention_pooling(WuQ_uQ, uQ, WvQVrQ, v0)[-1]
            print('Shape of rQ: {}'.format(rQ.get_shape()))

        # PointerNet
        with tf.variable_scope('pointer_network'):
            # initial state
            state = rQ

            Wha = tf.get_variable('Wha', [2 * h_size, h_size], dtype=tf.float32, initializer=xavier_initializer())
            WhP = tf.get_variable('WhP', [2 * h_size, h_size], dtype=tf.float32, initializer=xavier_initializer())
            v1 = tf.get_variable('v1', [h_size], dtype=tf.float32, initializer=xavier_initializer())
            gru_cell = tf.contrib.rnn.GRUCell(2 * h_size)

            processed_memory = mat_weight_mul(hP, WhP)
            processed_query = tf.expand_dims(tf.matmul(state, Wha), 1)
            s_0, _, c_0 = attention_pooling(processed_memory, hP, processed_query, v1)

            state, _ = gru_cell.call(c_0, state)
            processed_query = tf.expand_dims(tf.matmul(state, Wha), 1)
            s_1, _, c_1 = attention_pooling(processed_memory, hP, processed_query, v1)

        with tf.variable_scope('loss'):
            loss_0 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(asi),
                                                             logits=s_0)
            loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(aei),
                                                             logits=s_1)

            loss = (loss_0 + loss_1) / 2.0

            p = tf.argmax(tf.stack([s_0, s_1], 1), axis=2)

        return loss, p

