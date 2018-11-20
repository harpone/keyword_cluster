import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision.datasets.mnist import read_label_file, read_image_file
import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os.path
import spacy
import errno
import torch
import json


class RedditDataset(data.Dataset):
    """Dataset consisting a Pandas dataframe with columns `subreddit` and `submission_title`,
    applying lemmatization etc on the fly (maybe not very efficient, but can also be used
    in test/eval phase and makes debugging easier).
    """

    def __init__(self, data_path=None, vocabulary_path=None, label_dict_path=None):
        """See `data_rotations` docstring.
        """
        df = pd.read_csv(data_path)
        self.xs = df.submission_title.values  # numpy array of strings
        self.ys = df.subreddit.values  # numpy array of strings
        vocabulary = json.loads(open(vocabulary_path).read())
        self.nlp = spacy.load('en')

        self.label_dict = json.loads(open(label_dict_path).read())
        self.vectorizer = CountVectorizer(vocabulary=vocabulary, dtype=np.int32)

    def __getitem__(self, index):
        """

        :param index: int
        :return: tuple x, y; x.shape = [len(vocabulary), ]; y=int
        """
        x = self.xs[index]
        y = self.ys[index]

        # transforms:
        y = self.label_dict[y]  # str to int
        x = self.lemmatize(x)
        x = np.asarray(self.vectorizer.fit_transform([x]).todense())[0]

        return x, y

    def __len__(self):

        return len(self.xs)

    def lemmatize(self, string):
        lst = []
        doc = self.nlp(string)
        for token in doc:
            if not token.is_stop and token.is_alpha and token.lemma_ != '-PRON-':  # TODO: fix, dirty!
                lst.append(token.lemma_)

        return ' '.join(lst)
