"""
Utilities to support exps
"""

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


def _random_seed_gen(size:int=100):
    np.random.seed(0)
    return np.random.choice(10000, size)

def _path(save_path, name, init_strategy, n_repeat, num_GP, n_iter, cluster_interval, acq, lr, train_iter, ucb_strategy):
    return f"{save_path}/OL-{name}-{init_strategy}-{acq}-R{n_repeat}-P{num_GP}-T{n_iter}_I{cluster_interval}_L{int(-np.log10(lr))}-TI{train_iter}-US{ucb_strategy}"

def save_res(save_path, name, res, n_repeat=2, num_GP=2, n_iter=40, init_strategy:str="kmeans", cluster_interval:int=1, acq:str='ts', ucb_strategy="exact", lr:float=1e-3, train_iter:int=10, verbose=True):
    file_path = _path(save_path, name, init_strategy, n_repeat, num_GP, n_iter, cluster_interval, acq, lr, train_iter, ucb_strategy)
    np.save(file_path, res)
    if verbose:
        print(f"File stored to {file_path}")

def load_res(save_path, name, n_repeat=2, num_GP=2, n_iter=40, init_strategy:str="kmeans",  cluster_interval:int=1, acq:str='ts', ucb_strategy="exact", lr:float=1e-3,  train_iter:int=10, verbose=True):
    file_path = _path(save_path, name, init_strategy, n_repeat, num_GP, n_iter, cluster_interval, acq, lr, train_iter, ucb_strategy)
    file_path = f"{file_path}.npy"
    data = np.load(file_path)
    if verbose:
        print(f"Data {data.shape()} loaded from {file_path}")
    return data


