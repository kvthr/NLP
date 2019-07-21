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

# import modules for encoder and decoder
from models.encoder import *
from models.decoder import *

print("Creating an encoder object...")
encoder = BiLSTMEncoder(
    vocab_size = 1024,
    embedding_dim=128,
    encoder_size=16,
    batch_size=4
)
print("Created an encoder object.")

encoder_hidden = [[tf.zeros((1, 16)), tf.zeros((1, 16))], [tf.zeros((1, 16)), tf.zeros((1, 16))]]
# print(encoder_hidden.shape)

inputs = tf.convert_to_tensor([2, 3, 4])
print(inputs.shape)
inputs = tf.reshape(inputs, (1, -1))

outputs,encoder_hidden = encoder(inputs, encoder_hidden)

print(type(outputs))
print(outputs.shape)