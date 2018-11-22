# Similarity search for Reddit topics

**WARNING**: this is very much a work in progress. The ELMo 
embeddings and the similarity search work fine, but I was having
lots of trouble gitting a decent sized dataset for the ResNet and
just didn't have enough time to optimize the data processing.
Also probably has lots of bugs!

I won't have time to work on this anymore this week (spent around
16 hours on it already), so I'll just leave it for The Judgement
as it is now...


## Rough idea:
The idea is to train a classifier to predict the subreddit based on the 
title text, and then to use the classifier's near-top 
layer as an embedding and to measure similarity between titles
by nearest neighbor distance in the embedding space.

Current classifier is a fully connected neural network, and it feeds
on lemmatized and count-vectorized strings (reddit post titles). 
There's also ELMo embeddings as a baseline.

I actually regret a bit not implementing a simple character 
level RNN, since the title strings are quite short and there's
lots of data, so the char-rnn classifier would probably be able
to learn pretty good embeddings, and it would require virtually
no preprocessing... now I'm having lots of slowness problems 
with tokenizing, lemmatizing and buildint the vocabulary for the
resnet classifier. 
 
# Instructions

## ResNet model:
* run `extract_image_posts.py` to get only posts containing 
links to images
* then `create_vocabulary.py` will dump the vocabulary and label dictionary as json files to project root
* then `train_classifier.py`
* then run `build_embedding_lookup.py`, which will include the 
embedding vectors in the dataset (NOT DONE YET)
* then run `find_similar.py` to find top-n most similar posts
(NOT DONE YET)

## ELMo model:
* run `extract_image_posts.py` to get only posts containing 
links to images
* run `build_elmo_embeddings.py`. This will concat the dataset
with the ELMo embeddings and save it
* run `find_similar_elmo.py`. This will find closest match to
the user given query based on the ELMo embedding vectors

# Notes
* the `RedditDataset` does lemmatization (with spacy) on the fly,
which can be a bit slow... the point with this is that 1) it makes
dataset issues easy to debug by seeing the actual text/labels e.g.
in the debugger and 2) the dataset's `.vectorize` method can be
used also in the nn search phase
* Warning: `allennlp` seems to screw up stuff, such as matplotlib
imports and it requires certain versions of other packages...
* ELMo embeddings are quite high dimensional (256 for small model), 
so I don't expect the NN search to be very good for similarity 
search...
* Instead of using the last layer, last activation as the embedding
in ELMo, e.g. avg pool or multiple layers might make sense