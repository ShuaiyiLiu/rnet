# TODO: add bias
import math
import tensorflow as tf
from .rnn_cells import mat_weight_mul, masked_softmax, GatedAttentionCell, GatedAttentionSelfMatchingCell, PointerGRUCell


class RNet:
    @staticmethod
    def random_weight(dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    @staticmethod
    def dropout_wrapped_grucell(hidden_size, in_keep_prob, name=None):
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
        self.WuQ = self.random_weight(2 * h, h, name='WuQ')
        self.WuP = self.random_weight(2 * h, h, name='WuP')
        self.WvP = self.random_weight(h, h, name='WvP')
        self.v = self.random_weight(h, 1, name='v')
        self.Wg = self.random_weight(4 * h, 4 * h, name='Wg')
        self.Wg2 = self.random_weight(2 * h, 2 * h, name='Wg')
        self.WvP_hat = self.random_weight(h, h, name='WvP_hat')
        self.WvQ = self.random_weight(1, h, name='WvQ')
        self.Wha = self.random_weight(2 * h, h, name='Wha')
        self.WhP = self.random_weight(2 * h, h, name='WhP')

    def build_model(self, it):
        options = self.options
        # placeholders
        eP = it['eP']
        eQ = it['eQ']
        asi = it['asi']
        aei = it['aei']

        print('Shape of eP: {}'.format(eP.get_shape()))
        print('Shape of eQ: {}'.format(eQ.get_shape()))
        print('Shape of asi: {}'.format(asi.get_shape()))
        print('Shape of aei: {}'.format(aei.get_shape()))


        # embeddings concatenation
        # TODO: add character level embeddings
        eQcQ = eQ
        ePcP = eP
        p_max_length = options['p_length']
        q_max_length = options['q_length']
        qlen = self.length(eQ)
        plen = self.length(eP)
        print(qlen)
        print(plen)
        p_mask = tf.sequence_mask(plen, p_max_length, dtype=tf.float32)
        q_mask = tf.sequence_mask(qlen, q_max_length, dtype=tf.float32)
        print(p_mask)
        print(q_mask)

        ## difference: # of GRU layers, dropout application
        h_size = options['h_size']
        in_keep_prob = options['in_keep_prob']
        with tf.variable_scope('encoding') as scope:
            # TODO: number of layers as parameter
            gru_cells_fw = tf.nn.rnn_cell.MultiRNNCell([self.dropout_wrapped_grucell(h_size, in_keep_prob)
                                                        for _ in range(3)])
            gru_cells_bw = tf.nn.rnn_cell.MultiRNNCell([self.dropout_wrapped_grucell(h_size, in_keep_prob)
                                                        for _ in range(3)])

            uQ_2, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cells_fw, gru_cells_bw, eQcQ, dtype=tf.float32, sequence_length=qlen, scope='context_encoding')
            tf.get_variable_scope().reuse_variables()

            uP_2, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cells_fw, gru_cells_bw, ePcP, dtype=tf.float32, sequence_length=plen, scope='context_encoding')
            uQ = tf.concat(uQ_2, 2)
            uP = tf.concat(uP_2, 2)
            uQ = tf.nn.dropout(uQ, in_keep_prob)
            uP = tf.nn.dropout(uP, in_keep_prob)
            print('Shape of uP: {}'.format(uP.get_shape()))
            print('Shape of uQ: {}'.format(uQ.get_shape()))

        # Question and passage matching
        # Note: it is not clear here if bi-rnn or rnn should be used
        with tf.variable_scope('attention_matching'):
            weights = {
                'WuQ': self.WuQ,
                'WuP': self.WuP,
                'WvP': self.WvP,
                'v': self.v,
                'Wg': self.Wg
            }

            attn_cells = GatedAttentionCell(h_size, weights, uQ, q_mask)
            vP, _ = tf.nn.dynamic_rnn(cell=attn_cells, dtype=tf.float32, inputs=uP)
            vP = tf.nn.dropout(vP, in_keep_prob)

            print('Shape of vP: {}'.format(vP.get_shape()))

        # self matching layer
        with tf.variable_scope('self_matching'):
            weights = {
                'WvP': self.WvP,
                'v': self.v,
                'WvP_hat': self.WvP_hat,
                'Wg2': self.Wg2
            }
            attn_sm_cells_fw = GatedAttentionSelfMatchingCell(h_size, weights, vP, p_mask)
            attn_sm_cells_bw = GatedAttentionSelfMatchingCell(h_size, weights, vP, p_mask)
            hP_2, _ = tf.nn.bidirectional_dynamic_rnn(attn_sm_cells_fw,
                                                      attn_sm_cells_bw,
                                                      inputs=vP,
                                                      sequence_length=plen,
                                                      dtype=tf.float32)
            hP = tf.concat(hP_2, 2)
            hP = tf.nn.dropout(hP, in_keep_prob)
            print('Shape of hP: {}'.format(hP.get_shape()))

        # output layer
        with tf.variable_scope('output_layer'):
            # quote in section 4.2: After the original self-matching layer of the passage,
            # we utilize bi-directional GRU to deeply integrate the matching
            # results before feeding them into answer pointer layer.
            gru_cells_fw2 = tf.contrib.rnn.GRUCell(h_size)
            gru_cells_bw2 = tf.contrib.rnn.GRUCell(h_size)
            gP_2, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cells_fw2, gru_cells_bw2, hP, dtype=tf.float32, sequence_length=plen, scope='deeply_integration')
            gP = tf.concat(gP_2, 2)

            # question pooling
            WuQ_uQ = mat_weight_mul(uQ, self.WuQ)  # batch_size x q_length x H
            tanh = tf.tanh(WuQ_uQ + self.WvQ)
            s = mat_weight_mul(tanh, self.v)
            a = masked_softmax(s, 1, tf.expand_dims(q_mask, 2))
            rQ = tf.reduce_sum(tf.multiply(a, uQ), 1)
            rQ = tf.nn.dropout(rQ, in_keep_prob)
            print('Shape of rQ: {}'.format(rQ.get_shape()))

            # PointerNet
            s = []
            pt = []
            a = []
            Whp_hP = mat_weight_mul(gP, self.WhP)
            htm1a = rQ
            output_cell = tf.contrib.rnn.GRUCell(2 * h_size)
            for i in range(2):
                Wha_htm1a = tf.expand_dims(tf.matmul(htm1a, self.Wha), 1)
                tanh = tf.tanh(Whp_hP + Wha_htm1a)
                st = mat_weight_mul(tanh, self.v)
                s.append(tf.squeeze(st))
                at = masked_softmax(st, 1, tf.expand_dims(p_mask, 2))
                a.append(at)
                pt.append(tf.argmax(at, 1))
                ct = tf.reduce_sum(tf.multiply(at, gP), 1)
                _, htm1a = output_cell.call(ct, htm1a)

            p = tf.concat(pt, 1)
            print(p)

        with tf.variable_scope('loss_accuracy'):
            as_loss = self.cross_entropy(tf.one_hot(tf.squeeze(asi), p_max_length), tf.squeeze(a[0]))
            ae_loss = self.cross_entropy(tf.one_hot(tf.squeeze(aei), p_max_length), tf.squeeze(a[1]))
            loss = (as_loss + ae_loss) / 2.0

            as_accu, _ = tf.metrics.accuracy(labels=tf.squeeze(asi), predictions=pt[0])
            ae_accu, _ = tf.metrics.accuracy(labels=tf.squeeze(aei), predictions=pt[1])

            accu = (ae_accu + as_accu) / 2.0


        return loss, p, accu

