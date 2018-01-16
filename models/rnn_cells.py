from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf


def mat_weight_mul(mat, weight):
    mat_shape = mat.get_shape().as_list()
    weight_shape = weight.get_shape().as_list()
    assert (mat_shape[-1] == weight_shape[0])
    mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
    mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
    return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])


def _maybe_mask_score(score, memory_sequence_length, score_mask_value=float("-inf")):
    if memory_sequence_length is None:
        return score
    score_mask = tf.sequence_mask(
        memory_sequence_length, maxlen=tf.shape(score)[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)


def attention_pooling(processed_memory, memory, processed_query, v):
    s_t = tf.reduce_sum(v * tf.tanh(processed_memory + processed_query), [2])

    # alignments, reshaped for broadcasting, batch_size x memory_depth x 1
    a_t = tf.expand_dims(tf.nn.softmax(s_t), 2)

    # attention-pooling vector, batch_size x memory_units
    c_t = tf.reduce_sum(a_t * memory, 1)

    return s_t, a_t, c_t


class GatedAttentionGRUCell(RNNCell):

    def __init__(self, num_units, memory, memory_units, input_units, incorporate_state=False, reuse=None):
        super(GatedAttentionGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._cell = tf.contrib.rnn.GRUCell(num_units)
        self.memory = memory

        # weights initialization
        with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
            self.W_mem = tf.get_variable(
                'W_mem', shape=[memory_units, num_units], dtype=tf.float32, initializer=xavier_initializer())

            self.W_input = tf.get_variable(
                'W_input', shape=[input_units, num_units], dtype=tf.float32, initializer=xavier_initializer())

            self.W_state = None
            if incorporate_state:
                self.W_state = tf.get_variable(
                    'W_state', shape=[num_units, num_units], dtype=tf.float32, initializer=xavier_initializer())

            self.W_g = tf.get_variable(
                'W_g', shape=[memory_units + input_units, memory_units + input_units],
                dtype=tf.float32, initializer=xavier_initializer())

            self.v = tf.get_variable('v_attnetion', shape=[num_units], dtype=tf.float32, initializer=xavier_initializer())

        # processed memory
        self.keys = mat_weight_mul(memory, self.W_mem)


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        with tf.variable_scope('attention_pool'):
            # compute processed query
            processed_query = tf.matmul(inputs, self.W_input)
            if self.W_state is not None:
                # incorporate processed state into query
                processed_query += tf.matmul(state, self.W_state)

            processed_query = tf.expand_dims(processed_query, 1)

            c_t = attention_pooling(self.keys, self.memory, processed_query, self.v)[-1]

            ct_extended = tf.concat([inputs, c_t], 1) # batch_size x (memory_units + input_units)

            g_t = tf.sigmoid(tf.matmul(ct_extended, self.W_g)) # batch_size x (memory_units + input_units)

            ct_extended_star = g_t * ct_extended

            return self._cell.call(ct_extended_star, state)


if __name__ == '__main__':
    # test masked softmax
    import numpy as np
    def np_masked_softmax(x, axis, mask):
        m = tf.reduce_max(x, axis=axis, keep_dims=True)
        e = tf.exp(x - m) * mask
        s = tf.reduce_sum(e, axis=axis, keep_dims=True)
        s = tf.clip_by_value(s, 1e-8, 1e8)
        return e / s

    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def cross_entropy(labels, predict, mask):
        return -tf.reduce_sum(labels * tf.log(predict - mask + 1.0), 1)

    x = tf.constant([[[1.0], [0.0], [0.0], [0.0]], [[1.0], [2.0], [2.0], [0.0]]])
    mask = tf.constant([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0]])

    p = tf.constant([[[1.0, 1.0, 2.0, 0], [1.0, 0.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                     [[1.0, 1.0, 2.0, 0], [2.0, 0.0, 1.0, 1.0], [2.0, 0, 0, 0], [0, 0, 0, 0]]])
    print(x)
    print(mask)
    print(p)
    l = length(p)
    print(l)


    p_mask = tf.sequence_mask(l, 4, dtype=tf.float32)

    sess = tf.Session()
    softmaxed = np_masked_softmax(x, 1, tf.expand_dims(mask, 2))
    print(sess.run(np_masked_softmax(x, 1, tf.expand_dims(mask, 2))))
    print(sess.run(l))
    print(sess.run(p_mask))

    print('test error calculation')
    asi = [[0], [2]]
    print(sess.run(tf.one_hot(tf.squeeze(asi), 4)))

    sa = tf.one_hot(tf.squeeze(asi), 4)
    a = [[[1.0], [0.0], [0.0], [0.0]], [[.0], [1.0], [0.0], [0.0]]]
    print(sess.run(sa))

    print(sess.run(tf.squeeze(softmaxed) - mask + 1.0))
    print(sess.run(cross_entropy(sa, tf.squeeze(softmaxed), mask)))


    as_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(asi),
                                                             logits=tf.squeeze(x))
    print(sess.run(as_loss))
