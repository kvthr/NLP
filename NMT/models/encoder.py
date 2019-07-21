#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

# model class for BiLSTMEncoder
class BiLSTMEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_size, batch_size):
        super(BiLSTMEncoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )

        if tf.test.is_gpu_available():
            self.forward_rnn = tf.keras.layers.CuDNNLSTM(
                    units=self.encoder_size,
                    return_sequences=True,
                    return_state=True,
                    recurrent_initializer='glorot_unifrom'
                )
            self.backward_rnn = tf.keras.layers.CuDNNLSTM(
                    units=self.encoder_size,
                    return_sequences=True,
                    return_state=True,
                    go_backwards=True,
                    recurrent_initializer='glorot_unifrom'
                )
        else:
            self.forward_rnn = tf.keras.layers.LSTM(
                units=self.encoder_size,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
            )
            self.backward_rnn = tf.keras.layers.LSTM(
                units=self.encoder_size,
                return_sequences=True,
                return_state=True,
                go_backwards=True,
                recurrent_initializer='glorot_uniform'
            )

    def call(self, inputs, hidden):
        """
        Arguments:
            inputs - input to the Bi-directional LSTM, shape - (batch_size, seq_len)
            hidden - list of hidden states both forward and backward
        """
        inputs = self.embedding(inputs)
        f_out, f_hidden, f_context = self.forward_rnn(inputs, hidden[0])
        b_out, b_hidden, b_context = self.forward_rnn(inputs, hidden[1])
        # the output is a list consisting
        # concatenated encoder outputs(forward and backward), 
        # context and hidden for both forward and backward LSTMs
        encoder_out = tf.concat([f_out, b_out], axis=-1)
        
        f_encoder_state = [f_hidden, f_context]
        b_encoder_state = [b_hidden, b_context]
        return encoder_out, [f_encoder_state, b_encoder_state]

    def initialize_hidden(self):
        # intialize the hidden context states of the encoder
        return [[tf.zeros((self.batch_size, self.encoder_size)) for i in range(2)],
                [tf.zeros((self.batch_size, self.encoder_size)) for i in range(2)]]