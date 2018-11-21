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
from sklearn.metrics import f1_score
import itertools
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from datasets import RedditDataset
from models import ResNetFC

data_path = '/mnt/TERA/Data/reddit_topics/img_reddits_processed.csv'
vocabulary_path = './vocabulary.json'
label_dict_path = './label_dict.json'

# TODO: check dataset balance. acc probably a bad measure of performance! Maybe balanced sampling
# TODO: maybe Dice score; check imbalance
# TODO: predict
# TODO: script for building embedding lookup table and testing with post input text
# TODO: CLI instead of hard coded args
# TODO: count vector normalization?


#### Hyperparameters:
run_name = 'testrun_smthn'
hidden_size = 256
layers = 10
lr = 3e-4
batch_size = 64
seed = 1  # numpy seed for train/val split

validation_size = 128  # assuming for now that entire validation set in one minibatch
max_iters = 10000
eval_every = 100

#### Dataset:
dataset = RedditDataset(data_path=data_path,
                        vocabulary_path=vocabulary_path,
                        label_dict_path=label_dict_path)

np.random.seed(seed)
dataset_size = len(dataset)
index = np.arange(dataset_size)
np.random.shuffle(index)
index_trn = index[:-validation_size]
index_val = index[-validation_size:]

sampler_trn = SubsetRandomSampler(index_trn)
sampler_val = SubsetRandomSampler(index_val)

loader_trn = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         sampler=sampler_trn,
                                         num_workers=8,
                                         pin_memory=True)

loader_val = torch.utils.data.DataLoader(dataset,
                                         batch_size=validation_size,
                                         sampler=sampler_val)

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
        for xs, ys in loader_trn:
            model.train()

            # To GPU:
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
                for xs_val, ys_val in loader_val:  # TODO: assert 1 minibatch or support for several mbs
                    model.eval()

                    # To GPU:
                    xs_val = xs_val.cuda()
                    ys_val = ys_val.cuda()

                    logits_val, _ = model(xs_val)

                    loss_val = F.cross_entropy(logits_val, ys_val)

                    # Accuracy:
                    preds_val = torch.argmax(logits_val, dim=1)
                    acc_val = (ys_val == preds_val).float().mean()

                    # F1-score:
                    f1_score = f1_score(preds_val.cpu().numpy(), ys_val.cpu().numpy(), average='micro')

                    writer.add_scalar('val/loss', loss_val, global_step)
                    writer.add_scalar('val/acc', acc_val, global_step)
                    writer.add_scalar('val/f1_score', f1_score, global_step)

                    # Save models checkpoints
                    torch.save(model.state_dict(), join(save_path, 'model.pth'))
                    torch.save(optimizer.state_dict(), join(save_path, 'optimizer.pth'))

            global_step += 1
            if global_step >= max_iters:
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print(f'\nTraining stopped at iter={global_step}.')
