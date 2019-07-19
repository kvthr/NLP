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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(
    description="script for training the NMT model"
)
# data and pre-processing arguments
parser.add_argument(
    '-src_path',
    type=str,
    help="source language file path"
)
parser.add_argument(
    '-tgt_path',
    type=str,
    help="target language file path"
)

# add training hyperparameters
parser.add_argument(
    '-epochs',
    type=int,
    default=10,
    help="number of training epochs"
)
parser.add_argument(
    '-batch_size',
    type=int,
    default=8,
    help="size of the training mini-batch"
)
parser.add_argument(
    '-optimizer',
    type=str,
    default='adam',
    help="optimizer to be used"
)

parser.add_argument(
    '-model_dir',
    type=str,
    help="directory to store model files"
)

parser.add_argument(
    '-log_dir',
    type=str,
    help="directory to store log/tensorboard files"
)

args = parser.parse_args()
checkpoint_dir = args.model_dir + '/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# create a writer to write summaries/losses
global_step = tf.train.get_or_create_global_step()
writer = tf.contrib.summary.create_file_writer(args.log_dir)

def run(args):
    print("Creating Langauge Indices for source and target...")
    src_LI = LanguageIndex(language="source")
    tgt_LI = LanguageIndex(language="target")

    src_LI.add(read_file(args.src_path))
    tgt_LI.add(read_file(args.tgt_path))
    print("Created Langauge Indices.")

    print("Instantiating DataLoader object...")
    Data = DataLoader(args.src_path, args.tgt_path, src_LI, tgt_LI, args.batch_size)
    print("Loaded data.")

    print("Creating an encoder object...")
    encoder = BiLSTMEncoder(
        vocab_size = len(src_LI.word2idx),
        embedding_dim=128,
        encoder_size=16,
        batch_size=args.batch_size
    )
    print("Created an encoder object.")

    print("Creating a decoder object...")
    decoder = Decoder(
        vocab_size = len(tgt_LI.word2idx),
        embedding_dim=128,
        decoder_size=16,
        batch_size=args.batch_size
    )
    print("Created a decoder object.")

    # create optimize
    if(args.optimizer=='adam'):
        optimizer = tf.train.AdamOptimizer()
    
    # function to calculate loss
    def loss_function(real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    # create a checkpoint object for saving the model
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

    for epoch in range(args.epochs):
        start = time.time()
        
        total_loss = 0
        num_batch = 0
        
        for (batch, (input_seq, target_seq)) in enumerate(Data.data):
            num_batch += 1
            loss = 0
            with tf.GradientTape() as tape:
                encoder_output = encoder(input_seq)
                # initialize decoder hidden state
                decoder_hidden = decoder.initialize_hidden()
                decoder_input = tf.expand_dims([tgt_LI.word2idx['<start>']] * args.batch_size, 1)       
                
                # Teacher forcing - feeding the target as the next input
                for t in range(1, target_seq.shape[1]):
                    # passing encoder_outputs to the decoder
                    predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden[0], encoder_output)
                    
                    loss += loss_function(target_seq[:, t], predictions)
                    
                    # using teacher forcing
                    decoder_input = tf.expand_dims(target_seq[:, t], 1)
            
            batch_loss = (loss / int(target_seq.shape[1]))
            # write batch_loss to the tensorboard logs
            with writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('TrainingLoss', batch_loss.numpy())
            total_loss += batch_loss
            
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            
            optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
        # saving (checkpoint) the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / num_batch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

if __name__=="__main__":
    run(args=args)