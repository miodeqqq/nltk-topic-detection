#! /usr/bin/env python

# -*- coding: utf-8 -*-

import nltk
from nltk import RegexpParser
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from tika import parser


def get_only_once_occured_words(tokens):
    """
    Get only those words that occured only once in tokenized data.
    """

    return FreqDist(tokens).hapaxes()


def remove_duplicated_words(tokens):
    """
    Remove duplicated words from tokenized data.
    """

    return sorted(set(tokens))


def remove_stopwords(tokens):
    """
    Remove stopwords (and, the, unless, about, etc.) from tokenized data.
    """

    return [t for t in tokens if t not in stopwords.words('english')]


def generate_plot(tokens):
    """
    Generate a chart of the 50 most frequent words in tokenized data, if length of word is greater than 4.
    """

    return FreqDist(word for word in tokens if len(word) > 4).plot(50, cumulative=True)


def find_long_words(tokens):
    """
    Get only words of greater length than 15.
    """

    return sorted([word for word in set(tokens) if len(word) > 15])


def find_long_and_common_words(tokens):
    """
    Words of greater length than 7 characters and a count of occurence greater than 7.
    Those words are the common content-bearing words of a text
    """

    return sorted([word for word in set(tokens) if len(word) > 7 and FreqDist(tokens)[word] > 7])


def get_10_most_frequent_words(tokens):
    """
    Get 10 most frequent words from tokenized data.
    """

    return FreqDist(word.lower() for word in tokens).most_common(10)


def normalize_word(word):
    """
    Convert into lower case.
    """

    return word.lower()


def convert_single_word_into_plural_form(word):
    """
    Converts single form of word into its plural form.
    """

    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'


def get_part_of_speech(tokens):
    """
    Get part of speech for tokenized data.
    """

    return [e for e in nltk.chunk.ne_chunk(nltk.pos_tag(tokens)) if type(e) is tuple]


def noun_phrase_chunking(part_of_speech_data):
    """
    Generate parse tree with given part of speech data as input.
    """

    grammar = r"""
        NP: {<DT|JJ|NN.*>+}
        PP: {<IN><NP>}
        VP: {<VB.*><NP|PP|CLAUSE>+$}
        CLAUSE: {<NP><VP>}
        """

    grammar2 = r"""
        NP: {<DT|NN>+} # Chunk sequences of NN and DT
        {<DT><JJ><NN>} # Chunk det+adj+noun
        """

    return RegexpParser(grammar).parse(part_of_speech_data).draw()


def extract_text_from_pdf(file):
    """
    Extract text from input PDF and tokenize it.
    """

    return RegexpTokenizer(r'\w+').tokenize(parser.from_file(file)['content'])
