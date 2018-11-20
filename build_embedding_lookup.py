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
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from models import ResNetFC


"""
Load a trained model, push dataset through it and save the embeddings together with the X, Y

"""