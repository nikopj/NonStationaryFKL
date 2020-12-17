import math
import torch
import gpytorch
import copy
import spectralgp
from gpytorch.kernels.kernel import Kernel
from torch.nn import ModuleList
from gpytorch.likelihoods import GaussianLikelihood

from ..means import LogRBFMean2D
from ..utils import spectral_init
from ..models import LatentGPModel2D
from ..priors import GaussianProcessPrior
from ..trainer import trainer

class NonStatSpectralGPKernel(Kernel):
	def __init__(self, omega=None, num_locs=20, normalize=False, **kwargs):
		super(NonStatSpectralGPKernel, self).__init__(**kwargs)
		self.register_parameter('latent_params', torch.nn.Parameter(torch.zeros(num_locs**2)))
		self.num_locs = num_locs

	def initialize_from_data(self, train_x, train_y, num_locs=20,
			latent_lh = None, latent_mod = None, period_factor = 8.,
			latent_mean=None, omega=None, **kwargs):
		if omega is None:
			x1 = train_x.unsqueeze(-1)
			x2 = train_x.unsqueeze(-1)
			tau = self.covar_dist(x1, x2, square_dist=False, diag=False, last_dim_is_batch=False)
			max_tau = torch.max(tau)
			max_tau = period_factor * max_tau
			omega1 = math.pi * 2. * torch.arange(self.num_locs).double().div(max_tau)
		self.register_parameter('omega1', torch.nn.Parameter(omega1))
		self.omega1.requires_grad = False
		self.dw = self.omega1[1]-self.omega1[0]
		W1, W2 = torch.meshgrid(omega1, omega1)
		omega = torch.stack([W1.flatten(), W2.flatten()]).t() # (num_locs, 2)
		self.register_parameter('omega', torch.nn.Parameter(omega))
		self.omega.requires_grad = False
		# initialize log-periodogram
		#log_periodogram = torch.ones(self.num_locs**2).double()
		log_periodogram = -(omega[:,0]-omega[:,1])**2 / (2*self.omega.max())
		#print("omega_grid.shape", omega_grid.shape)
		#print("log_periodogram.shape", log_periodogram.shape)
		# if latent model is passed in, use that
		if latent_lh is None:
			self.latent_lh = GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3))
		else:
			print("Using specified latent likelihood")
			self.latent_lh = latent_lh
		#update the training data to include this set of omega and log_periodogram
		if latent_mod is None:
			if latent_mean is None:
				print("Using LogRBF2D latent mean")
				latent_mean = LogRBFMean2D
			self.latent_mod = LatentGPModel2D(omega, log_periodogram, self.latent_lh, mean=latent_mean)
		else:
			print("Using specified latent model")
			self.latent_mod = latent_mod
			self.latent_mod.set_train_data(omega, log_periodogram, strict=False)
		# set the latent g to be the de-meaned periodogram
		# and make it not require a gradient (we're using ESS for it)
		self.latent_params.data = log_periodogram
		self.latent_params.requires_grad = False
		# clear cache and reset training data
		self.latent_mod.set_train_data(inputs=omega, targets=self.latent_params.data, strict=False)
		# register prior for latent_params as latent mod
		latent_prior = GaussianProcessPrior(self.latent_mod, self.latent_lh)
		self.register_prior('latent_prior', latent_prior, lambda: self.latent_params)
		self.W1, self.W2 = W1, W2
		return self.latent_lh, self.latent_mod

	def forward(self,x1,x2,diag=False,last_dim_is_batch=False, **kwargs):
		S = torch.exp(self.latent_params) # density
		Sgrid = S.reshape(1,1,self.num_locs, self.num_locs)
		# reshape variables for broadcasting
		x1 = x1.reshape(-1,1,1,1)
		x2 = x2.reshape(1,-1,1,1)
		W1 = self.W1.reshape(1,1,*self.W1.shape)
		W2 = self.W2.reshape(1,1,*self.W2.shape)
		# (N,N,Nw,Nw)
		integrand = 0.25*Sgrid*(torch.cos(W1*(x1-x2)) + torch.cos(W1*x1-W2*x2) + torch.cos(W2*x1-W1*x2) + torch.cos(W2*(x1-x2)))
		# normalization constant
		Z = torch.trapz(torch.trapz(Sgrid.squeeze(), dx=self.dw), dx=self.dw)
		# kernel (N, N)
		output =  torch.trapz(torch.trapz(integrand, dx=self.dw), dx=self.dw) / Z
		if diag:
			return output.diag()
		return output

	def get_latent_mod(self, idx=None):
		return self.latent_mod

	def get_latent_lh(self, idx=None):
		return self.latent_lh

	def get_omega(self, idx=None):
		return self.omega

	def get_latent_params(self, idx=None):
		return self.latent_params

	def set_latent_params(self, g, idx=None):
		self.latent_params.data = g
		






