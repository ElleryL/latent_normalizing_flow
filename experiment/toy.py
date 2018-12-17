import random
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

seed = 14
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

DTYPE = tf.float32
NP_DTYPE = np.float32
D = 2



class Toy():
    def __init__(self):
        pass

    def mixture_circle(self):
        x1 = np.random.uniform(size=[80, 1]) * 2 * np.pi
        x2 = np.random.uniform(size=[120, 1]) * 2 * np.pi
        x3 = np.random.uniform(size=[40, 1]) * 2 * np.pi
        x4 = np.random.uniform(size=[160, 1]) * 2 * np.pi
        x5 = np.random.uniform(size=[50, 1]) * 2 * np.pi

        x5 = np.concatenate((np.cos(x5), np.sin(x5)), 1)
        x1 = np.concatenate((np.cos(x1) + 5., np.sin(x1) - 5.), 1)
        x2 = np.concatenate((np.cos(x2) + 5., np.sin(x2) + 5.), 1)
        x3 = np.concatenate((np.cos(x3) - 5., np.sin(x3) + 5.), 1)
        x4 = np.concatenate((np.cos(x4) - 5., np.sin(x4) - 5.), 1)

        X = np.concatenate((x1, x2, x3, x4, x5), 0)
        np.random.shuffle(X)
        return X.astype(NP_DTYPE)

    def mixture_gaussian(self):
        x1 = np.random.multivariate_normal(mean=[5., 5.], cov=np.diag(np.ones(2)) * 1, size=100)
        x2 = np.random.multivariate_normal(mean=[-5., -5.], cov=np.diag(np.ones(2)) * 1, size=100)
        x3 = np.random.multivariate_normal(mean=[5., -5.], cov=np.diag(np.ones(2)) * 1, size=100)
        x4 = np.random.multivariate_normal(mean=[-5., 5.], cov=np.diag(np.ones(2)) * 1, size=100)

        X = np.concatenate((x1, x2, x3, x4), 0)
        np.random.shuffle(X)
        return X.astype(NP_DTYPE)