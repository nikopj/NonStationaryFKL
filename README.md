# Functional Kernel Learning (FKL)

This repository contains a GPyTorch implementation of functional kernel learning (FKL) from the paper,

[Function-Space Distributions over Kernels](http://papers.nips.cc/paper/9634-function-space-distributions-over-kernels).

We've contributed to the original author's work to have a latent Gaussian Process that 
induces a distribution over *non-stationary* kernels
in the spectral domain, instead of only stationary kernels. This implementation
is for the 1D regression case and can be used via the `regression_runner.py`
interface (see argument `--stationary`) in the experiments folder (`exps/`). 
Preliminary results (such as in stationary kernel recovery) are promising in
that they suggest the proposed non-stationary FKL model can be used as a
drop-in replacement to the original model -- not merely complementary in
non-stationary regimes.

This work was completed as a final project in Professor Wilson's Bayesian Machine Learning Course at NYU. 



