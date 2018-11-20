import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import json
import spacy


min_posts = 100
n_keywords = 10

path_root = '/mnt/TERA/Data/reddit_topics'
path_img_data = join(path_root, 'img_reddits.csv')

print('Loading data...')
df = pd.read_csv(path_img_data)

df = df[['subreddit', 'submission_title']]
df_orig = df.copy()


nlp = spacy.load('en')


def lemmatizer(string):
    lst = []
    doc = nlp(string)
    for token in doc:
        if not token.is_stop and token.is_alpha and token.lemma_ != '-PRON-':  # TODO: fix, dirty!
            lst.append(token.lemma_)

    return lst

print('Lemmatizing...')
submission_titles = df['submission_title'].apply(lemmatizer)  # 1 min for 10k sentences!!
df['submission_title'] = submission_titles


# only subreddits with > min_posts posts:
top_subreddits = df['subreddit'].loc[(df['subreddit'].value_counts() > min_posts).values].unique()
print(top_subreddits)

df_top = df.loc[df.subreddit.isin(top_subreddits)]
df_orig = df_orig.loc[df_orig.subreddit.isin(top_subreddits)]


# top N most common keywords per subreddit:

top_kws = df_top.groupby('subreddit').sum()


# Collect top words per subreddit and total:

def count_words(lst_of_strs, top_n=10):
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
label_dict = {label: n for n, label in enumerate(top_subreddits)}

with open('label_dict.json', 'w') as f:
    json.dump(label_dict, f)

print('done.')

