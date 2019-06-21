#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import re

# from tensorflow examples
def preprocess_sentence(sentence=None):
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    
    sentence = sentence.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    sentence = '<start> ' + sentence + ' <end>'
    return sentence

def read_file(filepath=None):
    """ Read the file and return all the lines in the file."""
    data = []
    if filepath is not None:
        for line in open(filepath):
            sent = line.strip()
            # preprocess the sentence
            sent = preprocess_sentence(sentence=sent)
            
            data.append(sent.split(" "))
        return data
    else:
        logger.error("No filepath provided.")
