
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

class BaseStrategy(object):
    """
    Abstract class for a base ML predictor

    Parameters
    ----------- 
    name: string
        Name of the method.
    
    Attributes
    ----------
    name: str
        Name of method.

    Methods
    -------
    fit(X, y, **kwargs)
        Fits parameters of model to data.
    predict(X, **kwargs)
        Use model to predict ddG values using X features.
    load_model(X, y, **kwargs)
        Load model parameters into model.
    save_model(path)
        Save model parameters to a path.
    
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):  # , full_name):
        self.name = name
        self.recording = False

    def __str__(self):
        return self.name


    def save_model(self, path):
        with open(os.path.join(path, f'{self.name}'), 'wb') as f: 
            pickle.dump(self, f)

class MenuStrategy(BaseStrategy):
    """
    Abstract class for a base ML predictor
    
    Attributes
    ----------
    name: str
        Name of active learning algo for record keeping.
    model: `GPBase`
        underlying model that follows specification for basic model.
    train: bool
        Boolean whether we should train underlying model on some data before running selection.
    iters: int
        number of iterations to train underlying model for during intialization. Only used if train is `self.train = True`.
    
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, model=None, train=False, iters=20):  # , full_name):
        self.model = model
        self.name = name
        self.train = train
        self.iters = iters

    def __str__(self):
        return self.name

    @abstractmethod
    def fit(self,X,y,**kwargs):
        """
        Fit underlying model to data X using ddG values y

        Args:
            :param torch.Tensor X: The training inputs.
            :param torch.Tensor y: The training targets.

        """
        self.model.fit(X,y,**kwargs)


    @abstractmethod
    def select(self, X, i, k, **kwargs):
        """
        Select batch of `k` sequences from `X`. 

        Args:
            :param torch.Tensor X: NxD matrix where N is the number of seqs and D is # of features.
            :param torch.Tensor i: N shaped vector of encoded fidelities. (TODO: this should be a kwarg)
            :param int k: number of sequences to select

        Return:
            :return: list of indices that were selected. 
            :rtype: list
            ---- **optional**: these return values can be None ----
            :return:  mean value of predictive posterior distribution obtained during selection.
            :rtype: `torch.Tensor`
            :return: stddev of predictive posterior distribution obtained during selection.
            :rtype: `torch.Tensor`
            :return:  instantiation of torch.distribution joint_mvn covariance of posterior. 
            :rtype: `torch.Tensor`
        """
        pass

    @abstractmethod
    def evaluate(self, X, i):
        """
        Evaluate internal model at X sequences with i fidliety. 

        Args:
            :param torch.Tensor X: NxD matrix where N is the number of seqs and D is # of features.
            :param torch.Tensor i: N shaped vector of encoded fidelities. (TODO: this should be a kwarg)

        """
        pass

    @abstractmethod
    def score(self,X, joint_mvn_mean, joint_mvn_covar):
        """
        Internally score seqs in X based on `joint_mvn_mean` and `joint_mvn_covar`
        This method allows for flexible scoring rule, like adding penalties for
        mutational distance, a specific mutation, etc.

        Args:
            :param torch.Tensor X: NxD matrix where N is the number of seqs and D is # of features.
            :param torch.Tensor joint_mvn_mean: N shaped vector of mean value from predicitve posterior
            :param torch.Tensor joint_mvn_covar:   NxD covariance.  NOTE: not sure if this should lazy version provided by Gpytorch
            or the evaluated  matrix as tensor. For now I put tensor.
         Return:
            :return: scores for each indv. 
            :rtype: list
            ---- **optional**: these return values can be None ----
            :return:  mean value of predictive posterior distribution obtained during selection.
            :rtype: `torch.Tensor`
            :return: stddev of predictive posterior distribution obtained during selection.
            :rtype: `torch.Tensor`
            
        """
        pass

    @abstractmethod
    def update(self,X,y,**kwargs):
        """
        Update internal model at `X` seqs with `y` observations
        """
        self.model.resume_training(X,y,**kwargs)

    @abstractmethod
    def save(self, base_path):
        with open(os.path.join(base_path, self.name),'wb') as f:
            pickle.dump(self,f)

class DKBO_OLP_MenuStrategy(MenuStrategy):
    '''
    DKBO-Online Partition inherting Menu Strategy

    Attributes
    ----------
    name: str
        Name of active learning algo for record keeping.
    model: `DKL_BO_OLP_Batch`
        underlying model.
    train: bool
        Boolean whether we should train underlying model on some data before running selection.
    iters: int
        number of iterations to train underlying model for during intialization. Only used if train is `self.train = True`.
    partition_strategy:
        strategy of the online partition. (kmeans-y, kmeans, linear_rc, gp_rc).
    n_init: int
        number of points to initialize GP.
    num_GP: int
        number of partitions/ local GPs.
    train_times: int
        training iteration for each GP.
    acq: str
        acquisition function (UCB, TS).
    verbose: bool
        if printing verbose information.
    lr: float
        learning rate of the model.
    ucb_strategy: str
        strategy of coordinating UCB from multiple local GPs (exact, max, sum).
    pretrained: bool
        if using pretrained model to initialize the feature extracter of the DKL.
    ae_loc:
        location of the pretrained Auto Encoder.
    '''

    def __init__(self, x_tensor:torch.tensor, y_tensor:torch.tensor, partition_strategy:str="kmeans-y", num_GP:int=3, 
                    train_times:int=10, acq:str="ts", verbose:bool=True, lr:float=1e-2, name:str="test", ucb_strategy:str="exact",
                    train:bool=True, pretrained:bool=False, ae_loc:str=None):
        '''
        Args:
            @x_tensor: init x
            @y_tensor: init y
            @partition_strategy: strategy for the partition.
            @num_GP: desired number of local GPs.
            @train_times: number of training iteration for the models.
            @acq: acquisition function choice.
            @verbose: if printing verbose info.
            @lr: learning rate of the model.
            @name: instance name.
            @ucb_strategy: strategy to coordinate multiple local UCBs.
            @train: if updating the model.
            @pretrained: if using pretrained model to initilize the NN's weight.
            @ae_loc: location of the pretrained Auto Encoder.
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.partition_strategy = partition_strategy
        self.num_GP = num_GP
        self.acq = acq
        self.verbose = verbose
        self.lr = lr
        self.ucb_strategy = ucb_strategy
        self.pretrained = pretrained
        
        super().__init__(name=name, model=None, train=train, iters=train_times)
        self.train_times = self.iters
        self.if_train = self.train

        # load pretrained model
        if self.pretrained:
            assert not (ae_loc is None)
            self.ae = AE(x_tensor, lr=1e-3)
            self.ae.load_state_dict(torch.load(ae_loc, map_location=self.device))
        else:
            self.ae = None

        # initialize the model and partitioning
        self.init_x = x_tensor
        self.init_y = y_tensor
        self.n_init = self.init_x.size(0)
        self._dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_times, low_dim=True)
        self._dkl.train_model()
        self._dkl.model.eval()


 
    def select(self, X:torch.tensor, i:int, k:int, **kwargs):
        """
        Select k points from the N*D search space X of the fidelity i.

        Args:
            @X: N * D matrix to select from.
            @i: fidelity number.
            @k: number of points to be picked from X

        Return:
            :return: list of indices that were selected. 
            :rtype: list
        """
        x_tensor = torch.cat([self.init_x, X]) # could result in duplication
        _data_size = x_tensor.size(0)
        _util_array = np.arange(_data_size)
        self.observed = np.zeros(_data_size)
        self.observed[:self.n_init] = 1

        # partition
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _observed_pred = self._dkl.likelihood(self._dkl.model(x_tensor.to(self.device)))
            _, ucb = _observed_pred.confidence_region()
        self.cluster_id = clustering_methods(x_tensor, ucb.to('cpu'), self.num_GP, self.partition_strategy, 
                                        dkl=True, pretrained_nn=self.ae, train_iter=self.train_times)
        self.cluster_id_init = self.cluster_id[self.observed==1]

        init_x_list, init_y_list, test_x_list = [], [], []
        self.cluster_filter_list, self.cluster_idx_list = [], []
        for idx in range(self.num_GP):
            cluster_filter = self.cluster_id == idx
            cluster_filter[:self.n_init] = False # avoid querying the initial pts since it is not necessarily in X
            if np.sum(cluster_filter) == 0:   # avoid empty class -- regress to single partition.
                cluster_filter[self.n_init:] = True 
            self.cluster_filter_list.append(cluster_filter)
            self.cluster_idx_list.append(_util_array[cluster_filter])
            test_x_list.append(x_tensor[cluster_filter])
            init_x_list.append(self.init_x)
            init_y_list.append(self.init_y.reshape([-1,1]))

        self.model = DK_BO_OLP_Batch(init_x_list, init_y_list, lr=self.lr, train_iter=self.train_times,
                        n_init=self.n_init, num_GP=self.num_GP, pretrained_nn=self.ae)

        # batched query
        acq = kwargs.get("acq", "ts")
        verbose = kwargs.get("verbose", False)
        candidate_indices = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indices = self.model.query(test_x_list=test_x_list, n_iter=k, acq=acq, verbose=verbose) # list of (model_idx, candidate_idx)
        
        # recover indices in original input X
        # Question: what if the initial pts are queried? --> avoid picking these n_init pts
        selected_idx_list = []
        for (_model_idx, _candidate_idx) in indices:
            _original_idx = self.cluster_idx_list[_model_idx][_candidate_idx] - self.n_init
            assert _candidate_idx >= 0 and _candidate_idx < _data_size
            selected_idx_list.append(_original_idx)
        
        return selected_idx_list

    def update(self,X,y,**kwargs):
        """
        Update internal model at `X` seqs with `y` observations
        """
        # update the partitions
        self.init_x = torch.cat([self.init_x, X])
        self.init_y = torch.cat([self.init_y, y])
        self.n_init = self.init_x.size(0)
        self._dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_times, low_dim=True)
        self._dkl.train_model()
        self._dkl.model.eval()