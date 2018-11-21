import torch
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from models import ResNetFC
from datasets import RedditDataset

# parameters:
batch_size = 128  # can be quite big, this is just for testing

"""
Load a trained model, push dataset through it and save the embeddings together with the X, Y

"""

save_path = 'results/testrun'

# Load model hyperparameters:
hparams = json.loads(open(save_path).read())

# Define and load model:
model = ResNetFC(input_size=hparams['input_size'],
                 hidden_size=hparams['hidden_size'],
                 embedding_size=hparams['embedding_size'],
                 layers=hparams['layers'],
                 output_size=hparams['output_size']).cuda()
model.load_state_dict(torch.load(join(save_path, 'model.pth')))

# Load dataset: (this will ensure we will process the data exactly as during training)
data_path = hparams['data_path']
dataset = RedditDataset(data_path=hparams['data_path'],
                        vocabulary_path=hparams['vocabulary_path'],
                        label_dict_path=hparams['label_dict_path'])
sampler = SequentialSampler(dataset)
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=sampler,
                                     num_workers=8)

# Loop through entire dataset in minibatches:  # TODO make sure that really correctly aligned!
# TODO: now includes both train and val set... final model version should prolly be trained with entire dataset
embeddings = []
for xs, ys in loader:
    xs = xs.cuda()
    _, embedding = model(xs)  # shape [batch_size, embedding_size]
    embedding = embedding.detach().cpu().numpy()
    embeddings.append(embedding)

embeddings = np.concatenate(embeddings, axis=0)  # [len(dataset), embedding_size]# TODO: catches the last different size mb?

# Load actual dataframe:
df = pd.read_csv(data_path)
df['embeddings'] = embedding

# Save embedded dataset:
df.to_csv(join(save_path, 'img_reddits_embedded.csv'), index=False)




