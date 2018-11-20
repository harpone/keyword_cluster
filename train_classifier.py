import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from os.path import join
import shutil
import matplotlib.pyplot as plt
import itertools
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from datasets import RedditDataset


data_path = '/mnt/TERA/Data/reddit_topics/img_reddits.csv'
vocabulary_path = './vocabulary.json'
label_dict_path = './label_dict.json'

dataset_trn = RedditDataset(data_path=data_path,
                            vocabulary_path=vocabulary_path,
                            label_dict_path=label_dict_path)



x, y = dataset_trn.__getitem__(0)

print()

