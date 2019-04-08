# one place for all pre-processing utilities
import random
import numpy as np

def get_count_distinct(corpus):
    """
    Function to get the number of distinct words in the corpus and also the words.

    Arguments:
    corpus - a list of sentences(also a list of strings(words)).

    Returns:
    V - number of distinct words(Vocabulary size).
    vocab_words - distict words in the vocabulary, sorted.
    """
    # get the list of all words in the corpus
    vocab_words = [sentence[idx] for sentence in corpus for idx in range(len(sentence))]
    # remove duplicates
    vocab_words = set(vocab_words)

    return len(vocab_words), sorted(list(vocab_words))

def get_vocab_dicts(vocab_size, vocab):
    """
    Helper dicts for word indexing.

    Arguments:
    vocab_size - size of the vocabulary.
    vocab - list of vocabulary.

    Returns:
    word2idx - dictionary mapping words to indices.
    idx2word - dictionary mapping indices to words.
    """
    assert vocab_size == len(vocab)

    word2idx = {}
    idx2word = {}

    for idx in range(vocab_size):
        word2idx[vocab[idx]] = idx
        idx2word[idx] = vocab[idx]

    return word2idx, idx2word

def generate_batch(corpus, word2idx, window_size=2, batch_size=8):
    """
    Generate samples to train a skip-gram model.

    Arguments:
    corpus - corpus - a list of sentences(also a list of strings(words)).
    window_size - size of the context window.
    word2idx - dict that maps words to indices.

    Returns:
    context_indices - indices for the words in the context of the center word. 
    center_index - index of the center word.
    """

    # placeholder for batch data
    context_indices = np.zeros((batch_size, ))
    center_indices = np.zeros((batch_size, 1))

    # randomly sample a sentence from the corpus
    for i in range(batch_size):
        sampled_sentence = random.choice(corpus)
        # sample context and center word from the sentence
        center_idx = random.randrange(window_size, len(sampled_sentence)-window_size)
        context_idx = random.randrange(center_idx-window_size, center_idx+window_size)
        # populate the placeholders
        context_indices[i] = context_idx
        center_indices[i] = [center_idx]


    return context_indices, center_indices