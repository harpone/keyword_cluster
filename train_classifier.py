import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from os.path import join
import json
import shutil
import matplotlib.pyplot as plt
import torch.nn.functional as F
import itertools
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from datasets import RedditDataset
from models import ResNetFC

data_path = '/mnt/TERA/Data/reddit_topics/img_reddits_processed.csv'
vocabulary_path = './vocabulary.json'
label_dict_path = './label_dict.json'

# TODO: create large dataset and train
# TODO: predict
# TODO: script for building embedding lookup table and testing with post input text
# TODO: validation... or just tons of data
# TODO: check dataset balance. acc probably a bad measure of performance! Maybe balanced sampling
# TODO: CLI instead of hard coded args
# TODO: count vector normalization?
# TODO: maybe Dice score; check imbalance
# TODO: validation set + loss, acc, dice


#### Hyperparameters:
run_name = 'testrun_smthn'
hidden_size = 256
layers = 10
lr = 3e-4
batch_size = 64

max_iters = 10000
eval_every = 100

#### Dataset:
dataset = RedditDataset(data_path=data_path,
                        vocabulary_path=vocabulary_path,
                        label_dict_path=label_dict_path)

loader = DataLoader(dataset,
                    shuffle=True,
                    batch_size=batch_size,
                    num_workers=8,
                    pin_memory=True)

#### Model:
model = ResNetFC(input_size=dataset.vocabulary_size,
                 hidden_size=hidden_size,
                 layers=layers,
                 output_size=dataset.num_labels).cuda()

#### Optimizers:
optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       betas=(0.9, 0.999))

# Load/save:
save_path = join('results', run_name)

# Instantiate SummaryWriter:
writer = SummaryWriter(save_path)

# Save hyperparameters:
hparams = dict(input_size=dataset.vocabulary_size,
               hidden_size=hidden_size,
               layers=layers,
               output_size=dataset.num_labels,
               data_path=data_path,
               vocabulary_path=vocabulary_path,
               label_dict_path=label_dict_path)
with open(join(save_path, 'hparams.json'), 'w') as f:
    json.dump(hparams, f)

global_step = 0
try:
    while True:
        for xs, ys in loader:
            model.train()

            # To GPU: TODO device agnostic? NO TIIIIIME!!!!
            xs = xs.cuda()
            ys = ys.cuda()

            logits, _ = model(xs)

            # Optimize:
            loss = F.cross_entropy(logits, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy:
            preds = torch.argmax(logits, dim=1)
            acc = (ys == preds).float().mean()

            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/acc', acc, global_step)

            print(f'\r iter: {global_step} loss={np.round(loss.item(), 4)}', end='')

            if global_step % eval_every == 0:
                model.eval()

                # hs_val = model(xs_val)  # [1024, 1, 5, 1]  # TODO

                # Save models checkpoints
                torch.save(model.state_dict(), join(save_path, 'model.pth'))
                torch.save(optimizer.state_dict(), join(save_path, 'optimizer.pth'))

            global_step += 1
            if global_step >= max_iters:
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print(f'\nTraining stopped at iter={global_step}.')
