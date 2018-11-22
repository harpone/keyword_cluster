import torch
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import json
from annoy import AnnoyIndex
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from allennlp.commands.elmo import ElmoEmbedder

from utils import Lemmatizer


"""
Load saved model, take user input and find similar reddit topics based on similarity in embedding space.
"""

save_path = 'results/testrun'
embeddings_path = join(save_path, 'img_reddits_elmo_embeddings.csv')
query = 'Behold, an image of a donkey riding a bicycle'
n_neighbors = 10

# Load model hyperparameters:
hparams = json.loads(open(save_path).read())

# Define and load model:
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
model = ElmoEmbedder(options_file, weight_file, 0)

# Load embeddings:
df = pd.read_csv(embeddings_path)
embeddings = df.embeddings  # [len(dataset), 256]

# Process query:
lemmatizer = Lemmatizer()
x_query = lemmatizer(query)


# Find nearest neighbors:
nn_index = AnnoyIndex(len(embeddings))
nn_index.build(10)
# TODO: maybe save and/or load if exists; check speed
nns, distances = nn_index.get_nns_by_vector(x_query, n_neighbors, search_k=-1, include_distances=True)  # TODO: check usage and shapes
df_nns = df[['subreddit', 'submission_title']].loc[nns]

print(df_nns.to_string())  # TODO: print distances?

