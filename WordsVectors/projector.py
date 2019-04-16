# import required libraries
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# import pre-processing utilities
from utils.preprocess import *

with open('nltk_reuters_corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

vocab_size, vocab = get_count_distinct(corpus)
word2idx, idx2word = get_vocab_dicts(vocab_size, vocab)
window_size = 2

tsv_filepth = "metadata.tsv"
with open(tsv_filepth, 'w+', encoding='utf-8') as f:
    for i in range(vocab_size):
        f.write(idx2word[i]+'\n')

embeddings = np.load("./embeddings.npy")

# tensorflow Placeholders
X_init = tf.placeholder(tf.float32, shape=(vocab_size, 128), name="embedding")
X = tf.Variable(X_init)

# initializer
init = tf.global_variables_initializer()

# start Tensorflow session
sess = tf.Session()

# instance of Saver, save the graph.
saver = tf.train.Saver()
writer = tf.summary.FileWriter("./projector", sess.graph)

sess.run(init, feed_dict={X_init: embeddings})

#Configure a Tensorflow Projector
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.metadata_path = tsv_filepth

#Write a projector_config
projector.visualize_embeddings(writer,config)

#save a checkpoint
saver.save(sess, './projector/model.ckpt', global_step = vocab_size)

#close the session
sess.close()
