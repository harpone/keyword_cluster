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

#### Hyperparameters:
run_name = 'testrun'
lr = 3e-4
batch_size = 64

max_iters = 10000
eval_every = 1000

#### Dataset:
dataset = RedditDataset(data_path=data_path,
                        vocabulary_path=vocabulary_path,
                        label_dict_path=label_dict_path)

loader = DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=4,
                    pin_memory=True)


#### Model:  # TODO


def loss_fn(xs_, ys_):  # TODO

    return


#### Optimizers:
optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       betas=(0.9, 0.99))

# Instantiate SummaryWriter:
writer = SummaryWriter(join('results', run_name))
global_step = 0
try:
    while True:
        for xs, ys in loader:
            model.train()

            hs = model(xs)  # logit

            # Optimize:
            loss = loss_fn(hs, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', loss, global_step)

            print(f'\r iter: {global_step} loss={np.round(loss.item(), 4)}', end='')

            if global_step % eval_every == 0:
                model.eval()

                hs_val = model(xs_val)  # [1024, 1, 5, 1]


            global_step += 1
            if global_step >= max_iters:
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print(f'\nTraining stopped at iter={global_step}.')
