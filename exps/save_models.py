import math
import torch
import numpy as np


def save_model_output(alt_sampler, data_mod, omega, dat_name,
					  last_samples=10):
	fpath = "./saved_outputs/"
	## save model ##
	fname = fpath + dat_name + "_model.pt"
	torch.save(data_mod.state_dict(), fname)

	## save samples ##
	fname = fpath + dat_name + "_samples.pt"
	torch.save(alt_sampler.gsampled[0][0, :, -10:].detach(), fname)

	## save omega ##
	fname = fpath + dat_name + "_omega.pt"
	torch.save(omega, fname)

	return

def load_model_output(dat_name, data_mod=None):
	fpath = "./saved_outputs/"
	if data_mod is not None:
		fname = fpath + dat_name + "_model.pt"
		sd = torch.load(fname, map_location=torch.device('cpu'))
		data_mod.load_state_dict(sd)
	## save samples ##
	fname = fpath + dat_name + "_samples.pt"
	samples = torch.load(fname)
	## save omega ##
	fname = fpath + dat_name + "_omega.pt"
	omega = torch.load(fname)
	return samples, data_mod, omega
	

