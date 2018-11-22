import torch
import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt
import json
import argparse
from annoy import AnnoyIndex
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from allennlp.commands.elmo import ElmoEmbedder

from utils import Lemmatizer

# TODO: this is pretty slow because reading the dataframe and instantiating ELMo each time... continuously running
# process would of course be a better on-line alternative


"""
Load saved model, take user input and find similar reddit topics based on similarity in embedding space.
"""


def main(query=None,
         n_neighbors=None,
         embeddings_path=None,
         save_path=None):
    #save_path = 'results/testrun_smthn'
    #embeddings_path = join(save_path, 'img_reddits_elmo_embeddings.csv')
    #query = 'Behold, an image of a donkey riding a bicycle'
    #n_neighbors = 10

    # Load model hyperparameters:
    #hparams = json.loads(open(join(save_path, 'hparams.json')).read())

    # Define and load model:
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    model = ElmoEmbedder(options_file, weight_file, 0)  # TODO: slow to load...

    # Load embeddings:
    df = pd.read_csv(embeddings_path)
    embeddings = df[df.columns[2:]].values  # [len(dataset), 256]

    # Process query:
    lemmatizer = Lemmatizer()
    query_lemmatized = lemmatizer(query)
    query_embedding = model.embed_sentence(query_lemmatized)[2, -1]  # [256, ]

    # Find nearest neighbors:
    annoy_index_path = join(save_path, 'annoy_index.ann')
    nn_index = AnnoyIndex(embeddings.shape[1])
    if not os.path.exists(annoy_index_path):  # create the index and save
        # damn I guess I need to loop over the dataset to add items... pretty fast though
        for i, emb in enumerate(embeddings):
            nn_index.add_item(i, emb)

        nn_index.build(10)
        nn_index.save(annoy_index_path)

    else:  # load pre-existing index
        nn_index.load(annoy_index_path)

    nns, distances = nn_index.get_nns_by_vector(query_embedding, n_neighbors, search_k=-1, include_distances=True)  # TODO: check usage and shapes
    df_nns = df[['subreddit', 'submission_title']].loc[nns]

    print(df_nns.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vocabulary for the resnet model')
    parser.add_argument('-query',
                        help='The acutal unicode query',
                        default='Behold, an image of a donkey riding a bicycle!!')
    parser.add_argument('-n_neighbors',
                        help='Number of nearest matches to print',
                        default=10)
    parser.add_argument('-embeddings_path',
                        help='Path to the embedding file, created with `build_elmo_embeddings.py`',
                        default='./results/elmo_embeddings.csv')
    parser.add_argument('-save_path',
                        help='Temp path for Annoy index',
                        default='./annoy_index')
    args = parser.parse_args()

    main(**vars(args))

