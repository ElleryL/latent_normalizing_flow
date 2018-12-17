import random

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

import util
from models import RealNVP
from experiment import toy
tfd = tfp.distributions
tfb = tfp.bijectors
seed = 4
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

DTYPE = tf.float32
NP_DTYPE = np.float32

def log_prob_x_given_z(flows,num_comp,x):
    N = x.shape[0]
    result = tf.zeros((N,1))
    for i in range(num_comp):
        result = tf.concat((result,tf.reshape(flows[i].log_prob(x),[-1,1])),1)
    return result[:,1:]

def log_prob_z_given_x(flows,num_comp,x,log_z):
    log_X_given_z = log_prob_x_given_z(flows,num_comp,x)
    num = log_X_given_z + log_z
    deno = tf.reduce_logsumexp(num,axis=1)
    return (num-deno[:,None]),log_X_given_z

def initialization(x,num_comp):
    inx = np.random.choice(np.arange(x.shape[0]),size=num_comp,replace=False)
    return x[inx]


def train_mixture(train_step,
                  num_comp,
                  dim,
                  hidden_layers,
                  num_bijectors,
                  num_masked,
                  x,
                  batch_size,
                  lr

):

    sess = tf.Session()

    x_placeholder = tf.placeholder(DTYPE, [batch_size, dim], name="data")
    log_z_placeholder = tf.placeholder(DTYPE, [num_comp], name="log_z")
    log_z_given_x_placeholder = tf.placeholder(DTYPE, [batch_size, num_comp], name="log_z_given_x")

    # initialize normalizing flow
    flows = []

    # initialize base
    # mu = initialization(x,num_comp)
    mu = np.array([[-5., -5], [5., 5], [5., -5], [-5., 5]], dtype=NP_DTYPE)

    # initialize log_z
    log_z = np.log([1 / num_comp] * num_comp, dtype=NP_DTYPE)

    for i in range(num_comp):

        base_dist = tfd.MultivariateNormalDiag(loc=tf.constant(mu[i], dtype=DTYPE),
                                               scale_diag=tf.ones(dim, dtype=DTYPE) * 1.)
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

    log_Z_given_x, log_X_given_z = log_prob_z_given_x(flows, num_comp, x_placeholder, log_z_placeholder)
    log_Z = tf.reduce_logsumexp(log_z_given_x_placeholder, axis=0) - tf.log(tf.constant(batch_size, dtype=DTYPE))

    loss = - tf.reduce_sum(tf.reduce_sum(
        (log_X_given_z + log_z_placeholder - log_z_given_x_placeholder) * tf.exp(log_z_given_x_placeholder), axis=1))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    sess.run(tf.global_variables_initializer())
    LOSS = []
    for i in tqdm(range(train_step)):
        # E-step
        log_z_given_x = sess.run(log_Z_given_x, feed_dict={x_placeholder: x, log_z_placeholder: log_z})

        # M-step
        _, train_loss = sess.run([train_op, loss], feed_dict={x_placeholder: x,
                                                              log_z_placeholder: log_z,
                                                              log_z_given_x_placeholder: log_z_given_x})
        log_z = sess.run(log_Z, feed_dict={x_placeholder: x, log_z_given_x_placeholder: log_z_given_x})
        LOSS.append(train_loss)

