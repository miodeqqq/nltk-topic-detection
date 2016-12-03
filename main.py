#! /usr/bin/env python

# -*- coding: utf-8 -*-

import os
import sys

from utils import *


class TopicDetection(object):
    def __init__(self, input_file):
        self.input_file = input_file

    def compute_something(self):
        pdf = extract_text_from_pdf(self.input_file)

        print('\nPDF contains {} words'.format(len(pdf)))

        print('\nTen most frequent words are:\n{}'.format(get_10_most_frequent_words(pdf)))

        print('\nWords longer than 15 characters:\n{}'.format(find_long_words(pdf)))

        print('\nThe topic may be familiar with those words:\n{}'.format(find_long_and_common_words(pdf)))

td = TopicDetection(input_file=os.path.join(sys.argv[1]))
td.compute_something()
