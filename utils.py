import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Lemmatizer:
    """English language lemmatizer with stopword removal. Also transforms to lower case.

    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        self.stopwords = set(stopwords.words('english'))

    def __call__(self, string):
        lst = []
        try:
            doc = word_tokenize(string)
        except TypeError:
            doc = word_tokenize(str(string))
        for token in doc:
            if token not in self.stopwords and token.isalpha():
                lemma = self.lemmatizer.lemmatize(token)
                lst.append(lemma.lower())

        return lst

