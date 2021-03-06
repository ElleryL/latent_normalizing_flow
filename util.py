import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors




def alternate_coupling(D):

    d = D // 2

    L = list(range(d, D)) + list(range(0, d))
    f = tfb.Permute(permutation=L)
    return f


def sample_from_mixture(cat,flows,N,sess,dim):
    inx = np.random.choice(np.arange(len(cat)),p=cat,size=N)
    samples = np.empty((1,dim))
    for i in range(len(flows)):
        n = len(np.where(inx==i)[0])
        x = sess.run(flows[i].sample(n))
        samples = np.concatenate((samples,x),0)
        plt.scatter(x[:,0],x[:,1])
    plt.show()
    plt.close()
    return samples[1:]


def total_num_param():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters