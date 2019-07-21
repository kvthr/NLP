#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import numpy as np
import tensorflow as tf

# model class for simple Decoder
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decoder_size, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_size = decoder_size
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )
        
        if tf.test.is_gpu_available(): 
            self.rnn = tf.keras.layers.CuDNNLSTM(
                units=self.decoder_size,
                return_sequences=True,
                return_state=True
            )
        else:
            self.rnn = tf.keras.layers.LSTM(
            units=self.decoder_size,
            return_sequences=True,
            return_state=True
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

        # for Attention, we are using Bahadanau's Additive Attention
        self.W1 = tf.keras.layers.Dense(self.decoder_size)
        self.W2 = tf.keras.layers.Dense(self.decoder_size)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs, decoder_state, encoder_output):
        """

        Parameters:
        inputs: Shape-(batch_size, 1), pass the input at only one time step
        decoder_state: list of length 2, Shape-(batch_size, hidden_size), hidden state of the decoder
        encoder_output: Shape-(batch_size, max_sequence_length, 2*hidden_size)
        """

        # expand hidden_state in the time axis
        print(decoder_state[0].shape)
        hidden_time_expanded = tf.expand_dims(decoder_state[0], 1)
        print(hidden_time_expanded.shape)

        # get score for each time step of encoder hidden
        # using W1 and W2 of Additive Attention model
        # score: Shape-(batch_size, max_sequence_length, 1)
        score = self.V(tf.nn.tanh(self.W1(encoder_output) + self.W2(hidden_time_expanded)))
        # calculate the attention score using softmax
        attention_weights = tf.nn.softmax(score, axis=1)
        # calculate the context vector by attention weights
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # calculate embeddings of the input to decoder
        inputs = self.embedding(inputs)
 
        # concatenate the context vector and inputs
        inputs = tf.concat([tf.expand_dims(context_vector, axis=1), inputs], axis=-1)

        output, decoder_hidden, decoder_context = self.rnn(inputs, decoder_state)
        output_ = tf.reshape(output, (-1, output.shape[2]))
        
        # get the logits for the predicted token
        logits = self.fc(output_)
        
        return logits, [decoder_hidden, decoder_context], attention_weights

    def initialize_hidden(self):
        return [tf.zeros((self.batch_size, self.decoder_size)) for i in range(2)]
