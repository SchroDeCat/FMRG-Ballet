import os
import math
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
batch_size = 4
max_cholesky_size = float("inf")  # Always use Cholesky

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

class TuRBO():
    def __init__(self, train_x, train_y, n_init:int=10, acqf="ts", batch_size = 1, verbose=True, 
                    num_restarts=2, raw_samples = 512, discrete=True, pretrained_nn=None, train_iter=10, learning_rate=1e-2):
        def obj_func(pts):
            diff = torch.abs(train_x[:, :pts.size(0)] - pts)
            index = torch.argmin(torch.sum(diff, dim=1))
            return train_y[index]
        self.maximum = train_y.max()
        self.dim = train_x.size(1)
        self.training_iterations = train_iter
        self.lr = learning_rate
        self.obj_func = obj_func
        self.test_x = train_x if not discrete else train_x.float()
        self.test_y = train_y if not discrete else train_y.float()
        # self.X_turbo = get_initial_points(dim, n_init)
        self.X_turbo = train_x[:n_init]
        self.Y_turbo = torch.tensor(
            [self.obj_func(x) for x in self.X_turbo], dtype=dtype, device=device
        ).unsqueeze(-1)
        # print(f"init max {self.Y_turbo.max()}")
        self.batch_size = batch_size
        self.state = TurboState(self.dim, batch_size=batch_size)
        self.acqf = acqf
        self.verbose = verbose
        self.discrete = discrete
        self.pretrained_nn = pretrained_nn
        if self.discrete:
            assert batch_size == 1
        self.NUM_RESTARTS = num_restarts
        self.RAW_SAMPLES = raw_samples
        self.N_CANDIDATES = min(5000, max(2000, 200 * self.dim))
        self.likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        if not self.discrete:
            self.covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0))
            )
        else:
            # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            #     gpytorch.kernels.ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim, 
            #                                               lengthscale_constraint=Interval(0.005, 4.0))),
            #     num_dims=self.dim, grid_size=1000)
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1)),
                num_dims=self.dim, grid_size=10)
            # self.covar_module = gpytorch.kernels.LinearKernel()
    
    def opt(self, max_iter:int=100):
        low_dim = False
        # print(self.verbose)
        iterator = tqdm(range(max_iter)) if self.verbose else range(max_iter)
        for _ in iterator:
            # print(f"i {i}")
            # if state.restart_triggered:  # Run until TuRBO converges
            #     break
            # Fit a GP model
            # print(f"Size {self.X_turbo.size()} {self.Y_turbo.size()}")
            # print(f"x {self.X_turbo} y {self.Y_turbo}")
            self.train_Y = (self.Y_turbo - self.Y_turbo.mean()) / self.Y_turbo.std()
            # print(f"x {self.X_turbo} y {self.train_Y}")
            class LargeFeatureExtractor(torch.nn.Sequential):
                def __init__(self, data_dim):
                    super(LargeFeatureExtractor, self).__init__()
                    self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
                    self.add_module('relu1', torch.nn.ReLU())
                    self.add_module('linear2', torch.nn.Linear(1000, 500))
                    self.add_module('relu2', torch.nn.ReLU())
                    self.add_module('linear3', torch.nn.Linear(500, 50))
                    # test if using higher dimensions could be better
                    self.add_module('relu3', torch.nn.ReLU())
                    self.add_module('linear4', torch.nn.Linear(50, 1))

            class GPRegressionModel(gpytorch.models.ExactGP):
                    def __init__(self, train_x, train_y, gp_likelihood, gp_feature_extractor):
                        super(GPRegressionModel, self).__init__(train_x, train_y, gp_likelihood)
                        self.feature_extractor = gp_feature_extractor
                        self.mean_module = gpytorch.means.ConstantMean()
                        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1)),
                            num_dims=1, grid_size=100)

                        # This module will scale the NN features so that they're nice values
                        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

                    def forward(self, x):
                        # We're first putting our data through a deep net (feature extractor)
                        self.projected_x = self.feature_extractor(x)
                        self.projected_x = self.scale_to_bounds(self.projected_x)  # Make the NN values "nice"

                        mean_x = self.mean_module(self.projected_x)
                        covar_x = self.covar_module(self.projected_x)
                        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            if not self.discrete:
                self.model = SingleTaskGP(self.X_turbo, self.train_Y, covar_module=self.covar_module, likelihood=self.likelihood)
            else:
                self.feature_extractor = LargeFeatureExtractor(self.dim)
                if not (self.pretrained_nn is None):
                    self.feature_extractor.load_state_dict(self.pretrained_nn.encoder.state_dict(), strict=False)
                self.model = GPRegressionModel(self.X_turbo.float(), self.train_Y.float().squeeze(), self.likelihood, self.feature_extractor)
            self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

            # Do the fitting and acquisition function optimization inside the Cholesky context
            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                # Fit the model
                if not self.discrete:
                    fit_gpytorch_model(self.mll)
                else:
                    self.optimizer = torch.optim.Adam([
                        {'params': self.model.feature_extractor.parameters()},
                        {'params': self.model.covar_module.parameters()},
                        {'params': self.model.mean_module.parameters()},
                        {'params': self.model.likelihood.parameters()},
                    ], lr=self.lr)
                    self.loss_func = lambda pred, y: -self.mll(pred, y)
                    def train(verbose=False):
                        iterator = tqdm(range(self.training_iterations)) if verbose else range(self.training_iterations)
                        for i in iterator:
                            # Zero backprop gradients
                            self.optimizer.zero_grad()
                            # Get output from model
                            self.output = self.model(self.X_turbo.float())
                            # Calc loss and backprop derivatives
                            self.loss = self.loss_func(self.output, self.train_Y.float().squeeze())
                            self.loss.backward()
                            self.optimizer.step()


                            self.model.train()
                            self.likelihood.train()
                        if verbose:
                            iterator.set_postfix(loss=self.loss.item())
                    
                    # train(self.verbose)
                    train(verbose=False)
                # iterator.set_description(f'ML (loss={self.loss:.4})')
            
                # Create a batch
                self.X_next = self.next_point() if self.discrete else self.generate_batch()
                self.X_next = self.X_next.reshape([self.batch_size,-1])
                # print(f"Next point {self.X_next}")

            self.Y_next = torch.tensor(
                [self.obj_func(x) for x in self.X_next], dtype=dtype, device=device
            ).unsqueeze(-1)

            # Update state
            self.update_state()

            # Append data
            self.X_turbo = torch.cat((self.X_turbo, self.X_next), dim=0)
            self.Y_turbo = torch.cat((self.Y_turbo, self.Y_next), dim=0)

            # Print current status
            if self.verbose:
                # print(
                #     f"{len(self.X_turbo)}) Best value: {self.state.best_value:.2e}, TR length: {self.state.length:.2e}"
                # )
                iterator.set_postfix_str(f"({len(self.X_turbo)}) Regret: {self.maximum - self.state.best_value:.2e}, TR length: {self.state.length:.2e}")
            self.regret = self.maximum - np.maximum.accumulate(self.Y_turbo)
              
    def update_state(self,):
        state=self.state
        Y_next=self.Y_next
        if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        state.best_value = max(state.best_value, max(Y_next).item())
        if state.length < state.length_min:
            state.restart_triggered = True
        
        self.state = state
        return state

    def generate_batch(self):
        state=self.state
        model=self.model
        X=self.X_turbo
        Y=self.train_Y
        batch_size=self.batch_size
        n_candidates=self.N_CANDIDATES
        num_restarts=self.NUM_RESTARTS
        raw_samples=self.RAW_SAMPLES
        acqf=self.acqf

        # print(acqf, self.acqf, X.size(), Y.size())
        assert acqf in ("ts", "ei")
        # assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "ts":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = (
                torch.rand(n_candidates, dim, dtype=dtype, device=device)
                <= prob_perturb
            )
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask        
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        elif acqf == "ei":
            ei = qExpectedImprovement(model, self.test_y.max(), maximize=True)
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next

    def next_point(self, method="love", return_idx=False):
        """
        Maximize acquisition function to find next point to query
        """
        # clear cache
        self.model.train()
        self.likelihood.train()

        # Set into eval mode
        self.model.eval()
        self.likelihood.eval()
        acq=self.acqf

        test_x = self.test_x.to(device)

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

