# one place for all pre-processing utilities

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
    Function to get helper dicts for word indexing.

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