import matplotlib.pyplot as plt
from os.path import join
import torch.nn as nn
import torch.nn.functional as F


class ResNetFC(nn.Module):
    """Fully Connected ResNet. Last layer is the logit.

    """

    # TODO: maybe not optimal to use the layer right before the logit as the embedding... maybe a bottleneck layer?
    # or multiple layers => hierarchical similarity
    def __init__(self,
                 input_size=1,
                 hidden_size=1,
                 embedding_size=1,
                 output_size=1,
                 layers=1,
                 activation=nn.ReLU()):

        super().__init__()

        self.pre_layer = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        res_layers = []
        for l in range(layers):
            res_layers.append(nn.Linear(hidden_size, hidden_size))
        self.res_layers = nn.ModuleList(res_layers)
        self.embedding_layer = nn.Linear(hidden_size, embedding_size)
        self.post_layer = nn.Linear(embedding_size, output_size)
        self.act = activation

    def forward(self, x):
        """

        :param x: shape [B, N]
        :return logit, h: h can be used as an embedding
        """

        h = x
        h = self.pre_layer(h) if self.pre_layer is not None else h
        h = self.act(h)
        for fn in self.res_layers:
            h = h + self.act(fn(h))
        embedding = self.embedding_layer(h)
        logit = self.post_layer(embedding)
        return logit, embedding