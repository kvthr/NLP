#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

class LanguageIndex(object):
    def __init__(self, language=None, word2idx=None):
        self.language = language

        if word2idx is not None:
            self.word2idx = word2idx
        else:
            self.word2idx = {}
            self.idx2word = {}

            # vocab initialization
            self.word2idx['<pad>'] = 0
            self.word2idx['<start>'] = 1
            self.word2idx['<end>'] = 2
            self.word2idx['<unk>'] = 3

        self.unk_id = self.word2idx['<unk>']
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def add(self, text=None):
        """Updates the vocabulary of the language.
        text: list of phrases to update.
        """
        for phrase in text:
            for word in phrase:
                if word not in self.word2idx:
                    index = self.word2idx[word] = len(self.word2idx)
                    self.idx2word[index] = word

    def __repr__(self):
        """Representation when printed."""
        return "Language with VocabularySize=[{}]".format(len(self.word2idx))
