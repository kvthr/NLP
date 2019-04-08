# import libraries
import numpy as np
import tensorflow as tf

# import pre-processing utilities
from utils.preprocess import *
from model import SkipGram

corpus = [["i", "like", "computer"], ["computers", "are", "dumb"], ["i", "am", "smart"], ["i", "am", "great"], ["computer", "is", "great"]]
print("Building model...")

vocab_size, vocab = get_count_distinct(corpus)
word2idx, idx2word = get_vocab_dicts(vocab_size, vocab)
window_size = 1

graph = tf.Graph()
with graph.as_default() as g:

    model = SkipGram(vocab_size=vocab_size, embedding_dim=8, window_size=window_size, batch_size=1, graph=g)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        global_step = max(sess.run(model.global_step), 1)

        for _ in range(global_step, 15):

            global_step = sess.run(model.global_step) + 1

            context_indices, center_idx = generate_sample(corpus, word2idx, window_size=window_size)
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                                    model.context_indices: np.array(context_indices).reshape(model.batch_size,), model.center_idx: np.array(center_idx).reshape(model.batch_size, 1)})

            print(loss)