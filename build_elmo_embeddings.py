import torch
import numpy as np
import pandas as pd
import os
import argparse
from os.path import join
import matplotlib.pyplot as plt
from allennlp.commands.elmo import ElmoEmbedder

from utils import Lemmatizer

"""
Load a trained model, push dataset through it and save the embeddings together with the X, Y

"""


def main(data_path=None,
         save_path=None,
         nrows=None,
         batch_size=None):
    #save_path = 'results/test'
    #data_path = '/mnt/TERA/Data/reddit_topics/img_reddits.csv'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load ELMo (small):
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    model = ElmoEmbedder(options_file, weight_file, 0)

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
    df.to_csv(join(save_path, 'elmo_embeddings.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build embedding database using ELMo top layer activations')
    parser.add_argument('-data_path',
                        help='path to the reddit dataset (CSV)',
                        default='./img_reddits.csv')
    parser.add_argument('-save_path',
                        help='path to where the embedding dataframe will be saved (as CSV)',
                        default='./results/')
    parser.add_argument('-nrows',
                        help='number of rows to use in the dataset (yeah, a bit crappy)',
                        default=100000)
    parser.add_argument('-batch_size',
                        help='batch size to use when processing data *on GPU*',
                        default=256)
    args = parser.parse_args()

    main(**vars(args))


