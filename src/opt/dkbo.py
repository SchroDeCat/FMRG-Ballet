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

from ..models import DKL
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


class DK_BO():
    def __init__(self, train_x, train_y, lr=0.01, n_init:int=10, regularize=True, dynamic_weight=False, verbose=False, max=None, fix_seed=True, train_iter=10):
        if fix_seed:
            # print(n_init+n_repeat*n_iter)
            # _seed = rep*n_iter + n_init
            _seed = 70
            torch.manual_seed(_seed)
            np.random.seed(_seed)
            random.seed(_seed)
            torch.cuda.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

        # scale input
        ROBUST = True
        ScalerClass = RobustScaler if ROBUST else StandardScaler
        self.scaler = ScalerClass().fit(train_x)
        train_x = self.scaler.transform(train_x)
        # init vars
        self.regularize = regularize
        self.verbose = verbose
        self.n_init = n_init
        self.n_neighbors = min(self.n_init, 10)
        self.Lambda = 1
        self.dynamic_weight = dynamic_weight
        self.train_x = torch.from_numpy(train_x).float()
        self.train_y = train_y
        self.train_iter = train_iter
        self.maximum = torch.max(self.train_y) if max==None else max
        self.init_x = self.train_x[:n_init]
        self.init_y = self.train_y[:n_init]
        self.lr = lr
        self.dkl = DKL(self.init_x, self.init_y.squeeze(), lr=self.lr,  n_iter=self.train_iter, low_dim=True)
        self.cuda = torch.cuda.is_available()
        # self.cuda = False
        # gpytorch.add_jitter(self.dkl.model.)

        self.train()

    
    def train(self):
        if self.regularize:
            self.dkl.train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False, verbose=self.verbose)
        else:
            self.dkl.train_model()
        

    def query(self, n_iter:int=10):
        self.regret = np.zeros(n_iter)
        iterator = tqdm.notebook.tqdm(range(n_iter))
        for i in iterator:
            candidate_idx = self.dkl.next_point(self.train_x, "ts", "love", return_idx=True)
            # print(self.init_x.size(),  self.train_x[candidate_idx].size())
            self.init_x = torch.cat([self.init_x, self.train_x[candidate_idx].reshape(1,-1)], dim=0)
            self.init_y = torch.cat([self.init_y, self.train_y[candidate_idx].reshape(1,-1)])
            # retrain
            self.dkl = DKL(self.init_x, self.init_y.squeeze(), lr=self.lr, n_iter=self.train_iter, low_dim=True)
            # self.dkl.train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False)
            self.train()
            # regret
            self.regret[i] = self.maximum - torch.max(self.init_y)
            if self.regret[i] < 1e-10:
                break
            iterator.set_postfix(loss=self.regret[i])

