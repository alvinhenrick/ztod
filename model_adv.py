# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn


class HAND(object):

    def __init__(self, features, labels, vocab_size, num_classes, max_sentence_num=30, max_sentence_length=30,
                 embedding_size=200, hidden_size=50):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_sentence_num = max_sentence_num
        self.max_sentence_length = max_sentence_length
        self.input_x = features
        self.input_y = labels

        word_embedded = self.word_embedding()
        sent_vec = self.sent2vec_with_attention(word_embedded)
        doc_vec = self.doc2vec_with_attention(sent_vec)
        out = self.classifier(doc_vec)

        self.out = out

    def word_embedding(self):
        word_embed_layer = tf.contrib.layers.embed_sequence(self.input_x, vocab_size=self.vocab_size,
                                                            embed_dim=self.embedding_size)
        return word_embed_layer

    def sent2vec_with_attention(self, word_embedded):
        with tf.name_scope("sent2vec"):
            word_embedded = tf.reshape(word_embedded, [-1, self.max_sentence_length, self.embedding_size])
            word_encoded = self.bidirectional_gru_encoder(word_embedded, name='word_encoder')
            sent_vec = self.attention_layer(word_encoded, name='word_attention')
            return sent_vec

    def doc2vec_with_attention(self, sent_vec):
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size * 2])
            doc_encoded = self.bidirectional_gru_encoder(sent_vec, name='sent_encoder')
            doc_vec = self.attention_layer(doc_encoded, name='sent_attention')
            return doc_vec

    def classifier(self, doc_vec):
        with tf.name_scope('classification'):
            out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)
            return out

    def bidirectional_gru_encoder(self, inputs, name):
        with tf.variable_scope(name):
            gru_cell_fw = rnn.GRUCell(self.hidden_size)
            gru_cell_bw = rnn.GRUCell(self.hidden_size)

            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell_fw,
                                                                                 cell_bw=gru_cell_bw,
                                                                                 inputs=inputs,
                                                                                 dtype=tf.float32)
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def attention_layer(self, inputs, name):
        with tf.variable_scope(name):
            att_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='att_context')
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, att_context), axis=2, keep_dims=True), axis=1)
            attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return attention_output
