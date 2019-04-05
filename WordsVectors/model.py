# import libraries
import math
import numpy as np
import tensorflow as tf

# import pre-processing utilities
from utils.preprocess import *

class SkipGram(object):
    """
    """
    def __init__(self, vocab_size, embedding_dim, window_size, batch_size, num_sampled=16, graph=None):
        # word vector properties
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # skip-gram model properties
        self.window_size = window_size
        self.batch_size = batch_size
        # number of negative samples to be sampled
        self.num_sampled = num_sampled

        self.graph = if graph is not None else tf.Graph()

        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                                initializer=tf.constant_initializer(0), trainable=False)

            # placeholders for context(inputs) and center(labels)
            self.context_indices = tf.placeholder(tf.int32, shape=[self.batch_size, 2*self.window_size - 1])
            self.center_idx = tf.placeholder(tf.int32, shape=[batch_size, 1])

            # embedding matrix
            self.embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))

            # nce parameters
            self.nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocab_size, self.embedding_dim], stddev = 1.0/math.sqrt(self.embedding_dim)))
            self.nce_biases = tf.Variable(tf.zeros[self.vocab_size])

            # TODO
            # self.forward()

            # TODO
            # training and optimizer

    def forward(self):

        with tf.variable_scope("Embedding_Layer"):
            self.embed = tf.nn.embedding_lookup(self.embeddings, self.context_indices)

        with tf.variable_scope("NCE_Loss"):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                                biases=self.nce_biases,
                                labels=self.center_idx,
                                inputs=self.embed,
                                num_sampled=self.num_sampled,
                                num_classes=self.vocab_size))



