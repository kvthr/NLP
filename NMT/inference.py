#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import re
import sys
import argparse
import os
import time

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
# import data preprocessing utilities
from utils.utils import *
from utils.language import *
from DataLoader import *
# import modules for encoder and decoder
from models.encoder import *
from models.decoder import *

print("Creating Langauge Indices for source and target...")
src_LI = LanguageIndex(language="source")
tgt_LI = LanguageIndex(language="target")

src_LI.add(read_file("./data/train.en"))
tgt_LI.add(read_file("./data/train.es"))
print("Created Langauge Indices.")

print("Creating an encoder object...")
encoder = BiLSTMEncoder(
    vocab_size = len(src_LI.word2idx),
    embedding_dim=128,
    encoder_size=16,
    batch_size=16
)
print("Created an encoder object.")

print("Creating a decoder object...")
decoder = Decoder(
    vocab_size = len(tgt_LI.word2idx),
    embedding_dim=128,
    decoder_size=16,
    batch_size=16
)
print("Created a decoder object.")

optimizer = tf.train.AdamOptimizer()

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint("./training_checkpoints"))

def translate(input_sentence):

    sentence = preprocess_sentence(input_sentence)
    inputs = [src_LI.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=50, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''
    print(inputs.shape)
    encoder_out = encoder(inputs)
    print(encoder_out.shape)

    decoder_hidden = [tf.zeros((1, decoder.decoder_size)) for i in range(2)]
    decoder_input = tf.expand_dims([tgt_LI.word2idx['<start>']], 0)

    for t in range(50):
        predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden[0], encoder_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += tgt_LI.idx2word[predicted_id] + ' '

        if tgt_LI.idx2word[predicted_id] == '<end>':
            return result, sentence
        
        # the predicted ID is fed back into the model
        decoder_input = tf.expand_dims([predicted_id], 0)
        
    return result, sentence