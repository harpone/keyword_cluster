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
from allennlp.modules.elmo import Elmo, batch_to_ids

from datasets import RedditDataset
from create_vocabulary import lemmatize

# parameters:
batch_size = 128  # can be quite big, this is just for testing
nrows = 100000  # debugging with a small number

"""
Load a trained model, push dataset through it and save the embeddings together with the X, Y

"""

save_path = 'results/testrun_smthn'
data_path = '/mnt/TERA/Data/reddit_topics/img_reddits.csv'

# Load ELMo:
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
model = ElmoEmbedder(options_file, weight_file, 0)

# Load hyperparameters:
hparams = json.loads(open(join(save_path, 'hparams.json')).read())

# Load dataset: (this will ensure we will process the data exactly as during training)
df = pd.read_csv(data_path, nrows=nrows)
df = df[['subreddit', 'submission_title']]

# Lemmatizer stuff:
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = set(stopwords.words('english'))

def lemmatize(string):  # nltk lemmatizer
    lst = []
    doc = word_tokenize(string)
    for token in doc:
        if token not in stopwords and token.isalpha():
            lemma = lemmatizer.lemmatize(token)
            lst.append(lemma.lower())

    return lst

embeddings = []
masks = []
idx_mbs = np.array_split(np.arange(len(df)), len(df) // batch_size)
for idx in idx_mbs:
    sentence_mb = list(df['submission_title'].loc[idx].values)  # list of strs
    sentence_mb = [lemmatize(sentence) for sentence in sentence_mb]
    activations, masks = model.batch_to_embeddings(sentence_mb)  # shape [batch_size, embedding_size]
    # pick last non-masked activation as the embedding:
    last_timesteps = np.sum(masks.cpu().numpy(), axis=1) - 1
    embedding = activations[:, 2, last_timesteps, :]  # TODO: working up to here

    embeddings.append(embedding)

embeddings = np.concatenate(embeddings, axis=0)  # [len(dataset), embedding_size]# TODO: catches the last different size mb?


df['embeddings'] = embedding

# Save embedded dataset:
df.to_csv(join(save_path, 'img_reddits_embedded.csv'), index=False)




