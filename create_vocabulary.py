import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import json
import argparse
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils import Lemmatizer

"""
NOTES:
- 1M first posts: number of subreddits with > 100 posts = 62
- 10M first posts: number of subreddits with > 100 posts = 273; about 500k posts
"""

def main(min_posts=None,
         n_keywords=None,
         nrows=None,
         data_path=None,
         out_filename=None):

    #path_root = '/mnt/TERA/Data/reddit_topics'
    #out_filename = 'img_reddits_processed_1M.csv'
    #data_path = join(path_root, 'img_reddits.csv')

    print('Loading data...')
    df = pd.read_csv(data_path, nrows=nrows)

    df = df[['subreddit', 'submission_title']]

    # only subreddits with > min_posts posts:
    top_subreddits = df['subreddit'].loc[(df['subreddit'].value_counts() > min_posts).values].unique()
    print(top_subreddits)
    df_top = df.loc[df.subreddit.isin(top_subreddits)]

    # Instatiate lemmatizer:
    lemmatizer = Lemmatizer()

    print('Lemmatizing...')
    submission_titles = df_top['submission_title'].apply(lemmatizer)
    df_top['submission_title'] = submission_titles

    # top N most common keywords per subreddit:
    top_kws = df_top.groupby('subreddit').sum()  # TODO slooow!

    # Collect top words per subreddit and total:
    def count_words(lst_of_strs, top_n=10):
        """

        :param lst_of_strs:
        :param top_n: top n keywords to keep
        :return:
        """
        # print(lst_of_strs)
        word_counts = dict()
        for word in lst_of_strs:
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1

        # Sort:
        word_counts = {word: word_counts[word] for word in sorted(word_counts, key=word_counts.get, reverse=True)}

        # top_n:
        word_counts = {k: word_counts[k] for k in list(word_counts)[:top_n]}

        return word_counts


    top_all_words = []
    top_subreddit_words = dict()

    print('Counting words...')
    for index, row in top_kws.iterrows():
        # print(row.values)
        cnts = count_words(row.values[0], top_n=n_keywords)
        top_subreddit_words[index] = cnts
        # print(cnts)
        for word, _ in cnts.items():
            if word not in top_all_words:
                top_all_words.append(word)


    # Count-vectorize input text:
    vocabulary = {word: k for k, word in enumerate(top_all_words)}

    # Save vocabulary as file:
    print('Saving vocabulary to file...')
    with open('vocabulary.json', 'w') as f:
        json.dump(vocabulary, f)

    print('Saving label dictionary to file...')
    df['subreddit'].loc[(df['subreddit'].value_counts() > min_posts).values].unique()
    label_dict = {label: n for n, label in enumerate(top_subreddits)}

    with open('label_dict.json', 'w') as f:
        json.dump(label_dict, f)

    print('Processing dataset...')
    df_top.to_csv(out_filename, index=False)

    print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vocabulary for the resnet model')
    parser.add_argument('-min_posts',
                        help='Minimum number of posts in subreddit to include the class in dataset',
                        default=100)
    parser.add_argument('-n_keywords',
                        help='Number of top keywords per subreddit',
                        default=10)
    parser.add_argument('-nrows',
                        help='number of rows to use in the dataset (yeah, a bit crappy)',
                        default=100000)
    parser.add_argument('-data_path',
                        help='Path to the dataset (CSV file)',
                        default='./img_reddits.csv')
    parser.add_argument('-out_filename',
                        help='Path to output processed dataset',
                        default='./img_reddits_processed_100k.csv')
    args = parser.parse_args()

    main(**vars(args))

