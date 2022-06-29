import gpytorch
import os
import random
import torch
import tqdm
import time
import matplotlib
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import itertools

from sparsemax import Sparsemax
from scipy.stats import ttest_ind
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, low_dim=False):
        super(BasicFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        # test if using higher dimensions could be better
        if low_dim:
            self.add_module('relu3', torch.nn.ReLU())
            self.add_module('linear4', torch.nn.Linear(50, 1))

class InverseBasicFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, low_dim=False):
        super(InverseBasicFeatureExtractor, self).__init__()
        if low_dim:
            self.add_module('linear4', torch.nn.Linear(1, 50))
            self.add_module('relu4', torch.nn.ReLU())
        
        self.add_module('linear3', torch.nn.Linear(50, 500))
        self.add_module('relu3', torch.nn.ReLU())

        self.add_module('linear2', torch.nn.Linear(500, 1000))
        self.add_module('relu2', torch.nn.ReLU())
        
        self.add_module('linear1', torch.nn.Linear(1000, data_dim))

class AE(torch.nn.Module):
    def __init__(self,  train_x, lr=1e-1):
        super().__init__()
        self.gpu = torch.cuda.is_available()
        self.train_x = train_x
        self.learning_rate = lr
        self.weight_decay = 1e-8
        self.data_dim = self.train_x.size(1)
        self.decoder = InverseBasicFeatureExtractor(data_dim=self.data_dim)
        self.encoder = BasicFeatureExtractor(data_dim=self.data_dim)
        # Validation using MSE Loss function
        self.loss_function = torch.nn.MSELoss()

        # Using an Adam Optimizer with lr = 0.1
        self.optimizer = torch.optim.Adam(self.parameters(),
                                    lr = self.learning_rate,
                                    weight_decay = self.weight_decay)
    
        
    def train_ae(self, epochs=20, batch_size=100, verbose=False):
        self.train()
        self.losses = []
        self.train_x = self.train_x[:(self.train_x.size(0)//batch_size) * batch_size] # clip for batch training
        iterator = tqdm.tqdm(range(epochs))
        for epoch in iterator:
            for begin in range(self.train_x.size(0)//batch_size):
                # Output of Autoencoder
                batch_input = self.train_x[begin*batch_size: (begin+1)*batch_size]
                reconstructed = self.forward(batch_input)
                if self.gpu:
                    batch_input = batch_input.cuda()
                # Calculating the loss function
                self.loss = self.loss_function(reconstructed, batch_input)
                
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

            if verbose:
                iterator.set_postfix(loss=self.loss.item())                
            # Storing the losses in a list for plotting
            self.losses.append(self.loss.item())

        self.eval()
  
    def forward(self, x):
        if self.gpu:
            self.cuda()
            x = x.cuda()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def validation(self, test_x):
        '''
        Check Recovery loss on test set
        '''
        self.eval()
        reconstructed = self.forward(test_x)
        self._test_loss = self.loss_function(reconstructed, test_x)
        return self._test_loss

