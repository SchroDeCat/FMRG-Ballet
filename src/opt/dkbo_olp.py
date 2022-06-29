"""
Class for DKBO-OLP
"""

import gpytorch
import os
import random
from sklearn import cluster
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

class DK_BO_OLP():
    """
    distinguish history
    """
    def __init__(self, init_x_list, init_y_list, train_x, train_y, lr=0.01, train_iter:int=10, n_init:int=10, regularize=False, 
                        dynamic_weight=False, verbose=False, max_val:float=None, num_GP:int=2, pretrained_nn=None):
        # scale input
        ROBUST = True
        ScalerClass = RobustScaler if ROBUST else StandardScaler
        # init vars
        self.lr = lr
        self.regularize = regularize
        self.verbose = verbose
        self.n_init = n_init // num_GP
        self.n_neighbors = min(self.n_init, 10)
        self.Lambda = 1
        self.num_GP = num_GP
        self.dynamic_weight = dynamic_weight
        self.train_x = train_x
        self.train_y = train_y
        self.train_x_list, self.train_y_list, self.init_x_list, self.init_y_list = [], [], init_x_list, init_y_list
        self.dkl_list = []
        self.maximum = max_val
        self.train_iter = train_iter
        self.pretrained_nn = pretrained_nn

        # print(f"init_y_max {torch.max(torch.cat(self.init_y_list))}")
        for idx in range(num_GP):
            self.train_x_list.append(torch.from_numpy(ScalerClass().fit_transform(train_x[idx])).float())
            self.train_y_list.append(train_y[idx])
            tmp_dkl = DKL(self.init_x_list[idx], self.init_y_list[idx].squeeze(), lr=self.lr, n_iter=self.train_iter, low_dim=True,  pretrained_nn=self.pretrained_nn)
            self.dkl_list.append(tmp_dkl)
            # print("init length scale", self.dkl_list[idx].model.covar_module.base_kernel.outputscale.item())

        self.cuda = torch.cuda.is_available()

        for idx in range(num_GP):
            self.train(idx)
    
    def train(self, idx):
        if self.regularize:
            self.dkl_list[idx].train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False, verbose=self.verbose)
        else:
            self.dkl_list[idx].train_model() 
            # self.dkl_list[idx].model.covar_module.base_kernel.outputscale=1
            # print(self.dkl_list[idx].model.covar_module.base_kernel.outputscale.item())

    def query(self, n_iter:int=10, acq="ts", verbose=False):
        self.acq = acq
        self.observed = []
        self.regret = np.zeros(n_iter)
        iterator = tqdm.tqdm(range(n_iter)) if verbose else range(n_iter)

        for i in iterator:
            candidate_idx_list, candidate_acq_values = [], []
            for idx in range(self.num_GP):
                candidate_idx_list.append(self.dkl_list[idx].next_point(self.train_x_list[idx], acq, "love", return_idx=True))
                candidate_acq_values.append(self.dkl_list[idx].acq_val[candidate_idx_list[-1]].to("cpu"))

            candidate_model_idx = np.argmax(candidate_acq_values)
            candidate_idx = candidate_idx_list[candidate_model_idx]
            self.observed.append([candidate_model_idx, candidate_idx])
            
            # update specific model
            self.init_x_list[candidate_model_idx] = torch.cat([self.init_x_list[candidate_model_idx], self.train_x_list[candidate_model_idx][candidate_idx].reshape(1,-1)], dim=0)
            self.init_y_list[candidate_model_idx] = torch.cat([self.init_y_list[candidate_model_idx], self.train_y_list[candidate_model_idx][candidate_idx].reshape(1,-1)])
            # retrain
            self.dkl_list[candidate_model_idx] = DKL(self.init_x_list[candidate_model_idx], self.init_y_list[candidate_model_idx].squeeze(), lr=self.lr, n_iter=self.train_iter, low_dim=True,  pretrained_nn=self.pretrained_nn)
            self.train(candidate_model_idx)
            # regret
            self.regret[i] = self.maximum - torch.max(torch.cat(self.init_y_list))
            if self.regret[i] < 1e-10:
                break
            if verbose:
                iterator.set_postfix(loss=self.regret[i])
    
    def ucb_func(self, method="max", **kwargs):
        """
        Supported Method: max, exact, sum
        - max: maximum over multiple UCBs from those local GPs
        - exact: exactly from the local GP it belongs to
        - sum: balance between max & exact.
        """
        x_tensor = kwargs.get("x_tensor", self.train_x)
        cluster_id = kwargs.get("cluster_id", None)
        ucb_slot = torch.zeros(x_tensor.size(0))
        for id, _dkl in enumerate(self.dkl_list):
            _dkl.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    _observed_pred = _dkl.likelihood(_dkl.model(x_tensor.to(DEVICE)))
                    _, _upper = _observed_pred.confidence_region()
            if method.lower() == "max":
                ucb = _upper if id == 0 else torch.max(ucb, _upper)
            elif method.lower() == "sum":
                ucb = _upper if id == 0 else ucb + _upper
            elif method.lower() == "exact":
                self.dkl_list[id].next_point(x_tensor[cluster_id == id], "ucb", "love", return_idx=False)
                ucb_slot[cluster_id == id] = self.dkl_list[id].acq_val.to("cpu")
            else:
                raise NotImplementedError(f"{method} not implemented.")
        if method.lower() == 'exact':
            ucb = ucb_slot
        return ucb.to('cpu')


class DK_BO_OLP_Batch(DK_BO_OLP):
    """
    Batched version of DK_BO_OLP and don't rely on precollected Y.
    
    Attributes
    ----------
    name: str
        Name of active learning algo for record keeping.
    dkl_list: List of DKL
        underlying models.
    n_init: int
        number of points to initialize GP.
    num_GP: int
        number of partitions/ local GPs.
    train_iter: int
        training iteration for each GP.
    acq: str
        acquisition function (UCB, TS).
    verbose: bool
        if printing verbose information.
    lr: float
        learning rate of the model.
    pretrained: bool
        if using pretrained model to initialize the feature extracter of the DKL.
    """

    def __init__(self, init_x_list, init_y_list, lr:float=0.01, train_iter:int=10, n_init:int=10,
                    verbose:bool=False, num_GP:int=2, pretrained_nn=None):
        # init vars
        self.lr = lr
        self.verbose = verbose
        self.n_init = n_init
        self.num_GP = num_GP
        self.test_x_list, self.init_x_list, self.init_y_list = [], init_x_list, init_y_list
        self.dkl_list = []
        self.train_iter = train_iter
        self.pretrained_nn = pretrained_nn

        # print(f"init_y_max {torch.max(torch.cat(self.init_y_list))}")

        # init lists
        for idx in range(num_GP):
            # self.test_x_list.append(torch.from_numpy(ScalerClass().fit_transform(test_x[idx])).float())
            tmp_dkl = DKL(self.init_x_list[idx], self.init_y_list[idx].squeeze(), lr=self.lr, n_iter=self.train_iter, low_dim=True,  pretrained_nn=self.pretrained_nn)
            self.dkl_list.append(tmp_dkl)

        self.cuda = torch.cuda.is_available()

        for idx in range(num_GP):
            self.train(idx)
    
    def train(self, idx):
        """
        Train a certain DKL model in the model list.
        """
        # if self.regularize:
        #     self.dkl_list[idx].train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False, verbose=self.verbose)
        # else:
        self.dkl_list[idx].train_model() 

    def query(self, test_x_list, n_iter:int=10, acq="ts", verbose=False):
        """
        Use hallucinated points to better choose a batch to evaluate.
        Scaler removed.

        Input:
        1. test_x: the search spaces of each partition
        2. n_iter: num to choose in the batch.
        3. acq: acqusition functions

        Return:
        A list of [partition_id, candidate_id]
        """
        # scale input
        self.test_x = test_x_list
        for idx in range(self.num_GP):
            self.test_x_list.append(test_x_list[idx].float())
        self.acq = acq
        self.observed = []
        iterator = tqdm.tqdm(range(n_iter)) if verbose else range(n_iter)

        for i in iterator:
            candidate_idx_list, candidate_acq_values = [], []
            for idx in range(self.num_GP):
                candidate_idx_list.append(self.dkl_list[idx].next_point(self.test_x_list[idx], acq, "love", return_idx=True))
                candidate_acq_values.append(self.dkl_list[idx].acq_val[candidate_idx_list[-1]].to("cpu"))

            candidate_model_idx = np.argmax(candidate_acq_values)
            candidate_idx = candidate_idx_list[candidate_model_idx]
            self.observed.append([candidate_model_idx, candidate_idx])
            
            # update specific model with hallucination
            self.init_x_list[candidate_model_idx] = torch.cat([self.init_x_list[candidate_model_idx], self.test_x_list[candidate_model_idx][candidate_idx].reshape(1,-1)], dim=0)
            __hallucination = self.dkl_list[candidate_model_idx].pure_predict(self.test_x_list[candidate_model_idx][candidate_idx].reshape(1,-1)).reshape(1,-1)
            self.init_y_list[candidate_model_idx] = torch.cat([self.init_y_list[candidate_model_idx], __hallucination])
            
            # retrain
            self.dkl_list[candidate_model_idx] = DKL(self.init_x_list[candidate_model_idx], self.init_y_list[candidate_model_idx].squeeze(), lr=self.lr, n_iter=self.train_iter, low_dim=True,  pretrained_nn=self.pretrained_nn)
            self.train(candidate_model_idx) # the original framework of batched query do not engage retraining, 
                                            # but we follows the recent study that empirically shows retraining provides better performance.

        return self.observed[-n_iter:]


    