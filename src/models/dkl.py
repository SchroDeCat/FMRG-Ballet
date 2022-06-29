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

from .sgld import SGLD
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

class DKL():

    def __init__(self, train_x, train_y, n_iter=2, lr=1e-6, output_scale=.7, low_dim=False, pretrained_nn=None, test_split=False, retrain_nn=True):
        self.training_iterations = n_iter
        self.lr = lr
        self.train_x = train_x
        self.train_y = train_y
        self._y = train_y
        self._x = train_x
        self.data_dim = train_x.size(-1)
        self.low_dim = low_dim
        self.cuda = torch.cuda.is_available()
        self.test_split = test_split
        self.output_scale = output_scale
        self.retrain_nn = retrain_nn
        # self.cuda = False
        # split the dataset
        total_size = train_y.size(0)
        if test_split:
            test_ratio = 0.2
            test_size = int(total_size * test_ratio)
            shuffle_filter = np.random.choice(total_size, total_size, replace=False)
            train_filter, test_filter = shuffle_filter[:-test_size], shuffle_filter[-test_size:]
            self.train_x = train_x[train_filter]
            self.train_y = train_y[train_filter]
            self.test_x = train_x[test_filter]
            self.test_y = train_y[test_filter]
            self._x = self.test_x.clone()
            self._y = self.test_y.clone()
            self._x_train = self.train_x.clone()
            self._y_train = self.train_y.clone()

        class LargeFeatureExtractor(torch.nn.Sequential):
            def __init__(self, data_dim, low_dim):
                super(LargeFeatureExtractor, self).__init__()
                self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
                self.add_module('relu1', torch.nn.ReLU())
                self.add_module('linear2', torch.nn.Linear(1000, 500))
                self.add_module('relu2', torch.nn.ReLU())
                self.add_module('linear3', torch.nn.Linear(500, 50))
                # test if using higher dimensions could be better
                if low_dim:
                    self.add_module('relu3', torch.nn.ReLU())
                    self.add_module('linear4', torch.nn.Linear(50, 1))
                else:
                    self.add_module('relu3', torch.nn.ReLU())
                    self.add_module('linear4', torch.nn.Linear(50, 10))

        self.feature_extractor = LargeFeatureExtractor(self.data_dim, self.low_dim)
        if not (pretrained_nn is None):
            # print(self.feature_extractor.state_dict())
            # print(pretrained_nn.state_dict())
            self.feature_extractor.load_state_dict(pretrained_nn.encoder.state_dict(), strict=False)
            # print(self.feature_extractor, pretrained_nn)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        class GPRegressionModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, gp_likelihood, gp_feature_extractor):
                    super(GPRegressionModel, self).__init__(train_x, train_y, gp_likelihood)
                    self.feature_extractor = gp_feature_extractor
                    self.mean_module = gpytorch.means.ConstantMean()
                    if low_dim:
                        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1), 
                            outputscale_constraint=gpytorch.constraints.Interval(0.7,1.0)),
                            num_dims=1, grid_size=100)
                    else:
                        self.covar_module = gpytorch.kernels.LinearKernel(num_dims=10)

                    # This module will scale the NN features so that they're nice values
                    self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

                def forward(self, x):
                    # We're first putting our data through a deep net (feature extractor)
                    self.projected_x = self.feature_extractor(x)
                    self.projected_x = self.scale_to_bounds(self.projected_x)  # Make the NN values "nice"

                    mean_x = self.mean_module(self.projected_x)
                    covar_x = self.covar_module(self.projected_x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
        self.model = GPRegressionModel(self.train_x, self.train_y, self.likelihood, self.feature_extractor)
        if low_dim:
            self.model.covar_module.base_kernel.outputscale = self.output_scale

        if self.cuda:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.train_x = self.train_x.cuda()
            self.train_y = self.train_y.cuda()
    
    def train_model(self, loss_type="nll", verbose=False, **kwargs):
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        record_mae = kwargs.get("record_mae", False) 
        # Record MAE within a single training
        if record_mae: 
            record_interval = kwargs.get("record_interval", 1)
            self.mae_record = np.zeros(self.training_iterations//record_interval)
            self._train_mae_record = np.zeros(self.training_iterations//record_interval)

        # Use the adam optimizer
        if self.retrain_nn:
            params = [
                {'params': self.model.feature_extractor.parameters()},
                {'params': self.model.covar_module.parameters()},
                {'params': self.model.mean_module.parameters()},
                {'params': self.model.likelihood.parameters()},
            ]
        else:
            params = [
                {'params': self.model.covar_module.parameters()},
                {'params': self.model.mean_module.parameters()},
                {'params': self.model.likelihood.parameters()},
            ]
        # self.optimizer = torch.optim.Adam(params, lr=self.lr)
        self.optimizer = SGLD(params, lr=self.lr)

        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        if loss_type.lower() == "nll":
            self.loss_func = lambda pred, y: -self.mll(pred, y)
        elif loss_type.lower() == "mse":
            tmp_loss = torch.nn.MSELoss()
            self.loss_func = lambda pred, y: tmp_loss(pred.mean, y)
        else:
            raise NotImplementedError(f"{loss_type} not implemented")

        def train(verbose=False):
            iterator = tqdm.tqdm(range(self.training_iterations)) if verbose else range(self.training_iterations)
            for i in iterator:
                # Zero backprop gradients
                self.optimizer.zero_grad()
                # Get output from model
                self.output = self.model(self.train_x)
                # Calc loss and backprop derivatives
                self.loss = self.loss_func(self.output, self.train_y)
                self.loss.backward()
                self.optimizer.step()

                if record_mae and (i + 1) % record_interval == 0:
                    self.mae_record[i//record_interval] = self.predict(self._x, self._y, verbose=False)
                    if self.test_split:
                        self._train_mae_record[i//record_interval] = self.predict(self._x_train, self._y_train, verbose=False)
                    else:
                        self._train_mae_record[i//record_interval] = self.mae_record[i//record_interval]
                    if verbose:
                        iterator.set_postfix({"NLL": self.loss.item(),
                                        "Test MAE":self.mae_record[i//record_interval], 
                                        "Train MAE": self._train_mae_record[i//record_interval] })
                    self.model.train()
                    self.likelihood.train()
                elif verbose:
                    iterator.set_postfix(loss=self.loss.item())
                # iterator.set_description(f'ML (loss={self.loss:.4})')

        train(verbose)

    def predict(self, test_x, test_y, verbose=True, return_pred=False):
        self.model.eval()
        self.likelihood.eval()
        if self.cuda:
            test_x, test_y = test_x.cuda(), test_y.cuda()
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            preds = self.model(test_x)
        MAE = torch.mean(torch.abs(preds.mean - test_y))
        if self.cuda:
            MAE = MAE.cpu()


        self.model.train()
        self.likelihood.train()
        if verbose:
            print('Test MAE: {}'.format(MAE))
    
        if return_pred:
            return MAE, preds.mean.cpu() if self.cuda else preds.mean
        else:
            return MAE
    
    def pure_predict(self, test_x,  return_pred=True):
        '''
        Different from predict, only predict value at given test_x and return the predicted mean by default
        '''
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            preds = self.model(test_x)
        self.model.train()
        self.likelihood.train()

        if return_pred:
            return preds.mean.cpu()

    def latent_space(self, input_x):
        if eval:
            self.model.eval()
            self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            latent_x = self.model.feature_extractor(input_x)
        return latent_x.cpu() if self.cuda else latent_x

    def train_model_kneighbor_collision(self, n_neighbors:int=10, Lambda=1e-1, rho=1e-4, 
                                        regularize=True, dynamic_weight=False, return_record=False, verbose=False):
        # Find optimal model hyperparameters
        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        self.likelihood.train()
        self.NNeighbors = NearestNeighbors(n_neighbors=n_neighbors)
        self.softmax = torch.nn.Softmax(dim=1)
        self.penalty_record = torch.zeros(self.training_iterations)
        self.mse_record = torch.zeros(self.training_iterations)
        self.nll_record = torch.zeros(self.training_iterations)

        # Use the adam optimizer

        self.optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters()},
            {'params': self.model.covar_module.parameters()},
            {'params': self.model.mean_module.parameters()},
            {'params': self.model.likelihood.parameters()},
        ], lr=self.lr)

        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.penalty = 0

        def train():
            iterator = tqdm.tqdm(range(self.training_iterations)) if verbose else range(self.training_iterations)
            for i in iterator:
                # Zero backprop gradients
                self.optimizer.zero_grad()
                
                # Get output from model
                self.output = self.model(self.train_x)


                # collision penalty
                self.latent_variable = self.model.projected_x
                latent_variable = self.latent_variable.cpu() if self.cuda else self.latent_variable
                zero_var = torch.zeros([1,1]).cuda() if self.cuda else torch.zeros([1,1])
                self.NNeighbors.fit(latent_variable.detach().numpy())
                tmp_penalty = torch.zeros(self.latent_variable.size(0)).cuda() if self.cuda else torch.zeros(self.latent_variable.size(0))
                # tmp_weights = torch.zeros(self.latent_variable.size(0)).cuda() if self.cuda else torch.zeros(self.latent_variable.size(0))
                tmp_weights = self.train_y.detach()
                # for label, var in zip(self.train_y, self.latent_variable):
                for var_id, label in enumerate(self.train_y):
                    var_val = self.latent_variable[var_id].cpu() if self.cuda else self.latent_variable[var_id]
                    # tmp_weights[var_id] = label.detach().cpu().numpy() if self.cuda else label.detach().numpy()
                    tmp_weights[var_id] = label.detach()
                    neighbors_dists, neighbors_indices = self.NNeighbors.kneighbors(var_val.reshape([1,-1]).detach().numpy(), n_neighbors)
                    # for nei_dist, nei_idx  in zip(neighbors_dists[0], neighbors_indices[0]):
                    for nei_dist, nei_idx  in zip(neighbors_dists, neighbors_indices):
                        # z_dist = torch.norm(torch.abs(var   - self.latent_variable[nei_idx])) # will cause double backward
                        z_dist = np.linalg.norm(nei_dist)
                        y_dist = torch.norm(torch.abs(label - self.train_y[nei_idx])) * Lambda
                        assert torch.sum(self.train_y.cpu() - self._y)== 0
                        tmp = torch.maximum(zero_var,  y_dist - z_dist) ** 2
                        tmp_penalty[var_id] = tmp.reshape([1,1]).cuda() if self.cuda else tmp.reshape([1,1])   
                        # self.penalty += tmp

                if dynamic_weight:
                    self.penalty_weights = self.softmax(tmp_weights.reshape([1,-1]))
                    tmp_penalty = torch.sum(self.penalty_weights * tmp_penalty)
                self.penalty = sum(tmp_penalty / self.output.variance)
                self.penalty = self.penalty * rho
                            

                # Calc loss and backprop derivatives
                # print(self.output.mean.size(), self.train_y.size())
                nll = -self.mll(self.output, self.train_y)
                penalty = self.penalty
                if verbose:
                    print(f"nll {nll}, penalty {penalty}")
                # loss
                if regularize:
                    self.loss = nll + penalty
                else:
                    self.loss = nll
                # self.loss = penalty
                self.loss.backward(retain_graph=True)
                if verbose:
                    iterator.set_postfix(loss=self.loss.item())
                self.optimizer.step()
                # iterator.set_description(f'ML (loss={self.loss:.4})')

                # keep records
                if return_record:
                    pred_mean = self.output.mean.cpu() if self.cuda else self.output.mean
                    mse = torch.sum((pred_mean - self._y) ** 2)
                    # print(mse.detach())
                    # print(nll.detach(), penalty.detach())
                    self.mse_record[i] = mse.detach()
                    self.nll_record[i] = nll.detach()
                    self.penalty_record[i] = penalty

        # %time train()
        train()
        if return_record:
            return self.penalty_record, self.mse_record, self.nll_record

    def next_point(self, test_x, acq="ts", method="love", return_idx=False):
        """
        Maximize acquisition function to find next point to query
        """
        # clear cache
        self.model.train()
        self.likelihood.train()

        # Set into eval mode
        self.model.eval()
        self.likelihood.eval()

        if self.cuda:
          test_x = test_x.cuda()

        if acq.lower() == "ts":
            if method.lower() == "love":
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(200):
                    # NEW FLAG FOR SAMPLING
                    with gpytorch.settings.fast_pred_samples():
                        # start_time = time.time()
                        samples = self.model(test_x).rsample()
                        # fast_sample_time_no_cache = time.time() - start_time
            elif method.lower() == "ciq":
                with torch.no_grad(), gpytorch.settings.ciq_samples(True), gpytorch.settings.num_contour_quadrature(10), gpytorch.settings.minres_tolerance(1e-4):
                        # start_time = time.time()
                        samples = self.likelihood(self.model(test_x)).rsample()
                        # fast_sample_time_no_cache = time.time() - start_time
            else:
                raise NotImplementedError(f"sampling method {method} not implemented")
            self.acq_val = samples

        elif acq.lower() == "ucb":
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(test_x))
                lower, upper = observed_pred.confidence_region()
            self.acq_val = upper
        else:
            raise NotImplementedError(f"acq {acq} not implemented")

        max_pts = torch.argmax(self.acq_val)
        candidate = test_x[max_pts]
        if return_idx:
            return max_pts
        else:
            return candidate
