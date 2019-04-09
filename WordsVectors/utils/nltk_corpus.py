# script to construct the corpus in the required format for training

# import required libraries
import pickle
import re
from nltk.corpus import brown
from preprocess import *

# construct the sentences using categories
corpus = brown.sents(categories=['news', 'editorial', 'reviews', 'government'])

# join the tokens to form a sentence
clean_sents = []
for sent in corpus:
    st = ""
    for word in sent:
        st += word.lower() + " "
    clean_sents.append(st)

# clean and tokenize the sentences in teh corpus
corpus_clean = []
for sent in clean_sents:
    corpus_clean.append(clean_str(sent))

# save the corpus in pickle file
with open('nltk_reuters_corpus.pkl', 'wb') as f:
    pickle.dump(corpus_clean, f)

