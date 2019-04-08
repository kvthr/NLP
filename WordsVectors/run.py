# import libraries
import numpy as np
import tensorflow as tf

# import pre-processing utilities
from utils.preprocess import *
from model import SkipGram

print("Building corpus using NLTK library...")

print("Building model...")

vocab_size, vocab = get_count_distinct(corpus)
word2idx, idx2word = get_vocab_dicts(vocab_size, vocab)
window_size = 1

graph = tf.Graph()
with graph.as_default() as g:

    model = SkipGram(vocab_size=vocab_size, embedding_dim=8, window_size=window_size, batch_size=2, graph=g)

    with tf.Session() as sess:

        writer = tf.summary.FileWriter("./logs")
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        global_step = max(sess.run(model.global_step), 1)

        for _ in range(global_step, 15):

            global_step = sess.run(model.global_step) + 1

            context_indices, center_indices = generate_batch(corpus, word2idx, window_size=window_size, batch_size=model.batch_size)
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                                    model.context_indices: context_indices, model.center_idx: center_indices})

            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
            writer.add_summary(loss_sum, global_step)

            print(loss)