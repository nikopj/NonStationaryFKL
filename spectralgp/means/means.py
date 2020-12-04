import torch
import gpytorch
from torch.nn.functional import softplus
from gpytorch.priors import NormalPrior, MultivariateNormalPrior

class LogRBFMean(gpytorch.means.Mean):
	"""
	Log of an RBF Kernel's spectral density
	"""
	def __init__(self, hypers = None):
		super(LogRBFMean, self).__init__()
		if hypers is not None:
			self.register_parameter(name="constant", parameter=torch.nn.Parameter(hypers[-5] + softplus(hypers[-3]).log()))
			self.register_parameter(name="lengthscale", parameter=torch.nn.Parameter(hypers[-4]))
		else:
			self.register_parameter(name="constant", parameter=torch.nn.Parameter(0. * torch.ones(1)))
			self.register_parameter(name="lengthscale", parameter=torch.nn.Parameter(-0.3*torch.ones(1)))

		# register prior
		self.register_prior(name='constant_prior', prior=NormalPrior(torch.zeros(1), 100.*torch.ones(1), transform=None),
			param_or_closure='constant')
		self.register_prior(name='lengthscale_prior', prior=NormalPrior(torch.zeros(1), 100.*torch.ones(1), transform=torch.nn.functional.softplus),
			param_or_closure='lengthscale')

	def set_pars(self, hypers):
		self.constant.data = hypers[-2]
		self.lengthscale.data = hypers[-1]

	def forward(self, input):
		# logrbf up to constants is: c - t^1 / 2l
		out = self.constant - input.pow(2).squeeze(-1) / (2 * (softplus(self.lengthscale.view(-1)) + 1e-7) )
		return out

class LogRBFMean2D(gpytorch.means.Mean):
	"""
	Log of an RBF Kernel's spectral density
	"""
	def __init__(self, hypers = None):
		super(LogRBFMean2D, self).__init__()
		if hypers is not None:
			self.register_parameter(name="constant", parameter=torch.nn.Parameter(hypers[-5] + softplus(hypers[-3]).log()))
			self.register_parameter(name="lengthscale", parameter=torch.nn.Parameter(hypers[-4]))
		else:
			self.register_parameter(name="constant", parameter=torch.nn.Parameter(0. * torch.ones(1)))
			self.register_parameter(name="lengthscale", parameter=torch.nn.Parameter(-0.3*torch.ones(1)))

		# register prior
		self.register_prior(name='constant_prior', prior=NormalPrior(torch.zeros(1), 100.*torch.ones(1), transform=None),
			param_or_closure='constant')
		self.register_prior(name='lengthscale_prior', prior=NormalPrior(torch.zeros(1), 100.*torch.ones(1), transform=torch.nn.functional.softplus),
			param_or_closure='lengthscale')

	def set_pars(self, hypers):
		self.constant.data = hypers[-2]
		self.lengthscale.data = hypers[-1]

	def forward(self, input):
		# logrbf up to constants is: c - t^1 / 2l
		out = self.constant - (input[:,0]-input[:,1]).pow(2).squeeze(-1) / (2 * (softplus(self.lengthscale.view(-1)) + 1e-7) )
		return out


#class LogRBFMean2D(gpytorch.means.Mean):
#	"""
#	Log of 2D RBF Kernel's spectral density, with diagonal length-scale matrix.
#	"""
#	def __init__(self, hypers = None):
#		super(LogRBFMean2D, self).__init__()
#		if hypers is not None:
#			self.register_parameter(name="constant", parameter=torch.nn.Parameter(hypers[-5] + softplus(hypers[-3]).log()))
#			self.register_parameter(name="lengthscale", parameter=torch.nn.Parameter(hypers[-4]))
#		else:
#			self.register_parameter(name="constant", parameter=torch.nn.Parameter(0. * torch.ones(2)))
#			self.register_parameter(name="lengthscale", parameter=torch.nn.Parameter(-0.3*torch.ones(2)))
#
#		# register prior
#		self.register_prior(name='constant_prior', 
#			prior = MultivariateNormalPrior(torch.zeros(2), covariance_matrix=100.*torch.eye(2), transform=None), 
#			param_or_closure='constant')
#		self.register_prior(name='lengthscale_prior', 
#			prior = MultivariateNormalPrior(torch.zeros(2), covariance_matrix=100.*torch.eye(2), transform=torch.nn.functional.softplus), 
#			param_or_closure='lengthscale')
#
#	def set_pars(self, hypers):
#		self.constant.data = hypers[-2]
#		self.lengthscale.data = hypers[-1]
#
#	def forward(self, input):
#		# logrbf up to constants is: c - t^2 / 2l
#		out = self.constant - input.pow(2).sum(dim=1).squeeze(-1) / (2 * (softplus(self.lengthscale.view(-1)) + 1e-7) )
#		return out

