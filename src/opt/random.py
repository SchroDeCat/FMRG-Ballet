
'''
Implementation for experiments on LLNL simulation env.
'''

import pandas as pd
import numpy as np
from sklearn import cluster
import torch
import sys
import os
import dill as pickle

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

from ..models import DKL, AE
from ..utils import save_res, load_res, clustering_methods
from .dkbo_olp import DK_BO_OLP, DK_BO_OLP_Batch
from .dkbo_ae import DK_BO_AE
from abc import ABCMeta, abstractmethod
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


class RandomOpt:
    def __init__(self, y, name:str, ):
        self.y = y
        self.max = y.squeeze().max()
        self.name = name
        
    
    def opt(self, horizon:int, repeat:int=10):
        self.horizon = horizon
        self.repeat = repeat
        self.regret = np.zeros([self.repeat, self.horizon])
        for i in range(self.repeat):
            self._selection = np.random.choice(self.y.squeeze().size(0), horizon, replace=True)
            self._ys = self.y.squeeze()[self._selection]
            # print(f"self._ys {self._ys.shape} self.y {self.y.shape} ")
            self.regret[i] = np.minimum.accumulate(self.max - self._ys)
    

    def store_results(self, dir:str):
        np.save(f"{self.__file_name(dir)}.npy", self.regret)
        pass

    def plot_results(self, dir:str):
        plt.plot(self.regret.mean(axis=0), label=self.name)
        plt.xlabel("iter")
        plt.title("simple_regret")
        plt.savefig(f"{self.__file_name(dir)}.png")
        pass

    def __file_name(self, dir):
        return f"{dir}/RandomOpt-{self.name}-h{self.horizon}-r{self.repeat}"