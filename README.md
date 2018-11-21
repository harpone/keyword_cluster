# Similarity search for Reddit topics

Idea is to train a classifier to predict the subreddit based on the 
title text, and then to use the classifier near-top 
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
embeddin vectors in the dataset (NOT DONE YET)
* then run `find_similar.py` to find top-n most similar posts
(NOT DONE YET)