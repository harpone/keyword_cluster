import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from os.path import join
import shutil
import matplotlib.pyplot as plt
import torch.nn.functional as F
import itertools
import json
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SequentialSampler

from models import ResNetFC
from datasets import RedditDataset

# parameters:
batch_size = 128

"""
Load a trained model, push dataset through it and save the embeddings together with the X, Y

"""

save_path = 'results/testrun'

# Load model hyperparameters:
hparams = json.loads(open(save_path).read())

# Define and load model:
model = ResNetFC(input_size=hparams['input_size'],
                 hidden_size=hparams['hidden_size'],
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

# Loop through entire dataset:  # TODO make sure that really in correct order!
hs_all = []
for xs, ys in loader:
    xs = xs.cuda()
    _, hs = model(xs)
    hs_all.append(hs.detach().cpu().numpy())

