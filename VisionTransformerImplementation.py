import copy

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
import ImplementTransformer


class PatchEmbedder():

    def __init__(self, d_patch):
        self.d_patch = d_patch

    def CreateEmbeddings(self, x):
        x = x.view(-1, (self.d_patch ^ 2 ) * x[1])
        return x

    def EmbeddingMatrix(self, d_patch, d_model):
        #x is a tensor

        pe = torch.zeroes(d_patch ^ 2, d_model)
        nn.init.xavier_uniform(pe)






