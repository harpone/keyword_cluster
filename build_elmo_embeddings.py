import torch
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from allennlp.commands.elmo import ElmoEmbedder

from utils import Lemmatizer

# parameters:
batch_size = 256  # can be quite big, this is just for testing
nrows = 1000000  # debugging with a small number

"""
Load a trained model, push dataset through it and save the embeddings together with the X, Y

"""

save_path = 'results/testrun_smthn'
data_path = '/mnt/TERA/Data/reddit_topics/img_reddits.csv'

# Load ELMo (small):
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
model = ElmoEmbedder(options_file, weight_file, 0)

# Load hyperparameters:
hparams = json.loads(open(join(save_path, 'hparams.json')).read())

# Load dataset: (this will ensure we will process the data exactly as during training)
df = pd.read_csv(data_path, nrows=nrows)
df = df[['subreddit', 'submission_title']]

# Instatiate lemmatizer:
lemmatizer = Lemmatizer()

embeddings = []
masks = []
idx_mbs = np.array_split(np.arange(len(df)), len(df) // batch_size)
for idx in idx_mbs:
    sentence_mb = list(df['submission_title'].loc[idx].values)  # list of strs
    sentence_mb = [lemmatizer(sentence) for sentence in sentence_mb]
    activations, masks = model.batch_to_embeddings(sentence_mb)  # shape [batch_size, embedding_size]
    # pick last non-masked activation as the embedding (may not be optimal):
    last_timesteps = np.sum(masks.cpu().numpy(), axis=1) - 1
    embedding = activations[np.arange(len(last_timesteps)), 2, last_timesteps, :]

    embeddings.append(embedding)

embeddings = np.concatenate(embeddings, axis=0)
df_emb = pd.DataFrame(embeddings)
df = pd.concat([df, df_emb], axis=1)

# Save embedded dataset:
df.to_csv(join(save_path, 'img_reddits_elmo_embeddings.csv'), index=False)


