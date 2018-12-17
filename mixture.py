import random

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

import util
from models import RealNVP

tfd = tfp.distributions
tfb = tfp.bijectors
seed = 4
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

DTYPE = tf.float32
NP_DTYPE = np.float32

def train_mixture(num_comp,
                  dim,
                  hidden_layers,
                  num_bijectors,
                  num_masked,
                  train_data

):
    base_dist = tfd.MultivariateNormalDiag(loc=tf.constant([0.] * dim, dtype=DTYPE),
                                           scale_diag=tf.ones(dim, dtype=DTYPE) * 1.)

    # initialize weight
    weight = tf.Variable(initial_value=[1/num_comp] * num_comp,dtype=DTYPE)

    # initialize normalizing flow
    flows = []
    for i in range(num_comp):
        bijectors = []
        for j in range(num_bijectors):
            bijectors.append(RealNVP.RealNVP(
                                            num_masked=num_masked,
                                            shift_and_log_scale_fn=RealNVP.real_nvp_spectral_template(
                                            hidden_layers=hidden_layers)))
            bijectors.append(util.alternate_coupling(dim))
        Chain = tfb.Chain(list(reversed(bijectors[:-1])))
        Q = tfd.TransformedDistribution(distribution=base_dist,
                                        bijector=Chain)
        flows.append(Q)

