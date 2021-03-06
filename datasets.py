import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import spacy
import torch
import json


class RedditDataset(data.Dataset):
    """Dataset consisting a Pandas dataframe with columns `subreddit` and `submission_title`,
    applying lemmatization etc on the fly. Maybe not very efficient, but can also be used
    in test/eval phase and makes debugging easier. Also using higher number of workers in the
    loader should mitigate preprocessing slowness.
    """

    def __init__(self, data_path=None, vocabulary_path=None, label_dict_path=None):
        """See `data_rotations` docstring.
        """
        df = pd.read_csv(data_path)
        self.xs = df.submission_title.values  # numpy array of strings
        self.ys = df.subreddit.values  # numpy array of strings
        vocabulary = json.loads(open(vocabulary_path).read())
        self.vocabulary_size = len(vocabulary)
        self.nlp = spacy.load('en')

        self.label_dict = json.loads(open(label_dict_path).read())
        self.num_labels = len(self.label_dict)
        self.vectorizer = CountVectorizer(vocabulary=vocabulary, dtype=np.int32)

    def __getitem__(self, index):
        """

        :param index: int
        :return tuple x, y: x.shape = [len(vocabulary), ]; y=int
        """
        x = self.xs[index]
        y = self.ys[index]

        # transforms:
        y = torch.tensor(self.label_dict[y], dtype=torch.int64)  # str to int
        x = self.vectorize(x)

        return x, y

    def __len__(self):
        return len(self.xs)

    def vectorize(self, string):
        """

        :param string: just a (unicode) string
        :return x: torch.tensor of shape [len(vocabulary), ]
        """
        lst = []
        doc = self.nlp(string)
        for token in doc:
            if not token.is_stop and token.is_alpha and token.lemma_ != '-PRON-':  # TODO: fix, dirty!
                lst.append(token.lemma_)

        x = ' '.join(lst)
        x = np.asarray(self.vectorizer.fit_transform([x]).todense())[0]
        x = torch.tensor(x, dtype=torch.float32)

        return x

