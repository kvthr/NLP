# import libraries
import math
import numpy as np
import tensorflow as tf

# import packages from this module
from utils.preprocess import *


class SkipGram(object):
    """
    Model object with all training parameters for Skip-gram.

    Methods:
    __init__ - constructor for the model object.
    forward - forward pass through the network.
    """

    def __init__(self, vocab_size, embedding_dim, window_size, batch_size, num_sampled=16, graph=None):
        """
        Constructor method for the Skip-gram model object.

        Arguments:
        vocab_size - vocabulary size of the corpus.
        embedding_dim - size of the word vectors to train.
        window_size - context window size.
        batch_size - size of the batch while training.
        num_sampled - number of negative samples in NCE
        graph - Tensorflow graph.
        """
        # word vector properties
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # skip-gram model properties
        self.window_size = window_size
        self.batch_size = batch_size
        # number of negative samples to be sampled
        self.num_sampled = num_sampled

        self.graph = graph if graph is not None else tf.Graph()

        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)

            # placeholders for context(inputs) and center(labels)
            self.context_indices = tf.placeholder(
                tf.int32, shape=[self.batch_size])
            self.center_idx = tf.placeholder(tf.int32, shape=[batch_size, 1])

            # embedding matrix
            self.embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))

            # nce parameters
            self.nce_weights = tf.Variable(
                tf.truncated_normal([self.vocab_size, self.embedding_dim], stddev=1.0/math.sqrt(self.embedding_dim)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # forward pass through the graph
            self.forward()

            # training and optimizer
            self.lr = 0.001
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
            # compute and apply gradients
            grads = self.optimizer.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            self.train_op = self.optimizer.apply_gradients(
                zip(gradients, variables), global_step=self.global_step)

    def forward(self):
        """
        Forward pass through the network.
        """
        with tf.variable_scope("Embedding_Layer"):
            self.embed = tf.nn.embedding_lookup(
                self.embeddings, self.context_indices)

        with tf.variable_scope("NCE_Loss"):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                               biases=self.nce_biases,
                               labels=self.center_idx,
                               inputs=self.embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocab_size))
