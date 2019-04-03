# one place for all pre-processing utilities
import random

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

def generate_sample(corpus, word2idx, window_size=2):
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

    context_words = []
    sampled_sentence = random.choice(corpus)
    center_idx = random.randrange(window_size, len(sampled_sentence)-window_size)

    for i in range(window_size):
        context_words.append(sampled_sentence[center_idx + i + 1])
        context_words.append(sampled_sentence[center_idx - i - 1])

    return [word2idx[word] for word in context_words], word2idx[sampled_sentence[center_idx]]