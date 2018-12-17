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