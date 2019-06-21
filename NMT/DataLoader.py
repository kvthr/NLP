#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import re
import sys

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from utils.utils import *

# DataLoader object for generating training/evaluation data
class DataLoader(object):
    """ Load the parallel data for NMT.
    Batchify and preprocess the data.
    """
    def __init__(self,
                source_filepath=None,
                target_filepath=None,
                src_LanguageIndex=None,
                tgt_LanguageIndex=None,
                batch_size=16,
                max_sequence_length=100):
        self.src_tensors = read_file(filepath=source_filepath)
        self.tgt_tensors = read_file(filepath=target_filepath)

        self.src = src_LanguageIndex
        self.tgt = tgt_LanguageIndex

        self.batch_size = batch_size

        self.src_tensors = [[self.src.word2idx[word] for word in sent] for sent in self.src_tensors]
        self.tgt_tensors = [[self.tgt.word2idx[word] for word in sent] for sent in self.tgt_tensors]

        # pad the source and target sequences to max_sequence_length 
        self.src_tensors = tf.keras.preprocessing.sequence.pad_sequences(
            self.src_tensors,
            maxlen=max_sequence_length,
            padding='post'
        )
        self.tgt_tensors = tf.keras.preprocessing.sequence.pad_sequences(
            self.tgt_tensors,
            maxlen=max_sequence_length,
            padding='post'
        )
        # load data from tf.data
        self.data = tf.data.Dataset.from_tensor_slices((self.src_tensors, self.tgt_tensors)).shuffle(buffer_size=1000)
        self.data = self.data.batch(self.batch_size, drop_remainder=True)

    def get_batch(self):
        return next(iter(self.data))

