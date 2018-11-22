import torch
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import json
from annoy import AnnoyIndex

from datasets import RedditDataset
from models import ResNetFC

"""
Load saved model, take user input and find similar reddit topics based on similarity in embedding space.
"""
# TODO: WARNING -- JUST A DRAFT, WORK IN PROGRESS!
# TODO: Annoy documentation is... annoying
# TODO: maybe keep process running and keep asking for user input instead of loading the embeddings every time?

save_path = 'results/testrun'
query = 'Behold, an image of a donkey riding a bicycle'
n_neighbors = 10

# Load model hyperparameters:
hparams = json.loads(open(save_path).read())

# Define and load model:
model = ResNetFC(input_size=hparams['input_size'],
                 hidden_size=hparams['hidden_size'],
                 embedding_size=hparams['embedding_size'],
                 layers=hparams['layers'],
                 output_size=hparams['output_size']).cuda()
model.load_state_dict(torch.load(join(save_path, 'model.pth')))

# Load dataset:
data_path = hparams['data_path']
dataset = RedditDataset(data_path=hparams['data_path'],
                        vocabulary_path=hparams['vocabulary_path'],
                        label_dict_path=hparams['label_dict_path'])

# Load embeddings:
df = pd.read_csv(data_path)
embeddings = df.embeddings  # [len(dataset), len(vocabulary)]

# Process query:
x_query = dataset.vectorize(query)  # [len(vocabulary)]

# Find nearest neighbors:
nn_index = AnnoyIndex(len(dataset))
nn_index.build(10)
# TODO: maybe save and/or load if exists; check speed
nns, distances = nn_index.get_nns_by_vector(x_query, n_neighbors, search_k=-1, include_distances=True)  # TODO: check usage and shapes
df_nns = df[['subreddit', 'submission_title']].loc[nns]

print(df_nns.to_string())  # TODO: print distances?

