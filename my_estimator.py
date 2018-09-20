import tensorflow as tf
from tensorflow.contrib import layers

from model import HAND


def han_model_fn(features, labels, mode):
    hand = HAND(features,
                labels,
                vocab_size=5000,
                num_classes=2,
                embedding_size=200,
                hidden_size=50)

    logits = hand.out

    predictions = {'class_ids': tf.argmax(input=logits, axis=1),
                   'probabilities': tf.nn.softmax(logits),
                   'logits': logits,
                   }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    actual = tf.argmax(hand.input_y, axis=1, name='actual')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(logits, actual), tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=hand.input_y,
                                                                  logits=logits,
                                                                  name='loss'))
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
