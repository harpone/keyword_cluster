# Similarity search for Reddit topics

Idea is to train a classifier to predict the subreddit based on the 
title text, and then to use the classifier's near-top 
layer as an embedding and to measure similarity between titles
by nearest neighbor distance in the embedding space.

Current classifier is a fully connected neural network, and it feeds
on lemmatized and count-vectorized strings (reddit post titles).
 Future plans involve 1) simple char-rnn classifier 2) using e.g.
 pretrained ELMo embeddings instead of training a classifier. 
 
# Instructions

So far no CLI args. See inside the scripts if you want to change
the arguments.

* spacy installation requires the english language pack, which can be downloaded by `python -m spacy download en`
* first run `extract_image_posts.py` to get only posts containing links to images
* then `create_vocabulary.py` will dump the vocabulary and label dictionary as json files to project root
* then `train_classifier.py`
* then run `build_embedding_lookup.py`, which will include the 
embedding vectors in the dataset (NOT DONE YET)
* then run `find_similar.py` to find top-n most similar posts
(NOT DONE YET)

# Notes
* the `RedditDataset` does lemmatization (with spacy) on the fly,
which can be a bit slow... the point with this is that 1) it makes
dataset issues easy to debug by seeing the actual text/labels e.g.
in the debugger and 2) the dataset's `.vectorize` method can be
used also in the nn search phase
* Warning: `allennlp` seems to screw up stuff, such as matplotlib
imports and it requires certain versions of other packages...
* ELMo embeddings are high dimensional (1024), so I don't expect
the NN search to be very good for similarity search...
* Instead of using the last layer, last activation as the embedding
in ELMo, e.g. avg pool or multiple layers might make sense