# coding=utf-8

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.keras.preprocessing import sequence

from model import HAND
from yelp import load_data

sentence_size = 30

x_train_variable, y_train, x_test_variable, y_test = load_data("yelp_academic_dataset_review.json", 30, 30)

x_train = sequence.pad_sequences(
    x_train_variable,
    maxlen=sentence_size,
    value=0)
x_test = sequence.pad_sequences(
    x_test_variable,
    maxlen=sentence_size,
    value=0)

x_len_train = np.array([min(len(x), sentence_size) for x in x_train_variable])
x_len_test = np.array([min(len(x), sentence_size) for x in x_test_variable])


def parser(x, y):
    return x, y


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train_variable))
    dataset = dataset.batch(100)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.batch(100)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def han_model_fn(features, labels, mode):
    hand = HAND(features,
                labels,
                vocab_size=159654,
                num_classes=5,
                embedding_size=200,
                hidden_size=50)

    predictions = {'class_ids': tf.argmax(input=hand.out, axis=1),
                   'probabilities': tf.nn.softmax(hand.out),
                   'logits': hand.out,
                   }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=hand.input_y,
                                                                  logits=hand.out,
                                                                  name='loss'))

    accuracy = tf.metrics.accuracy(tf.argmax(hand.input_y, axis=1), predictions['class_ids'], name='accuracy')

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={'accuracy': accuracy})

    train_op = layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01,
        summaries=['loss', 'learning_rate'])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )
