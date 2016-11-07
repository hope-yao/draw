import chainer.serializers
chainer.serializers.load_npz('mnist_generator',generator)
# chainer.serializers.load_npz('mnist_discriminator',discriminator)
###### you could directly use the generator and discriminator after loading

# import argparse
# import numpy as np
# from chainer import datasets, cuda, serializers, Variable
# from chainer import optimizers as O
# from chainer import functions as F
# from models import Generator, Discriminator
#
#
# def rnd_categorical(n, n_categorical):
#     indices = np.random.randint(n_categorical, size=n)
#     one_hot = np.zeros((n, n_categorical))
#     one_hot[np.arange(n), indices] = 1
#     return one_hot, indices
#
#
# def rnd_continuous(n, n_continuous, mu=0, std=1):
#     return np.random.normal(mu, std, size=(n, n_continuous))
#
#
# n_z = 62
# n_categorical = 10
# n_continuous = 2
# max_epochs = 10000
# batch_size = 1000
#
#
# # Quick check the training result
# import random
# from matplotlib import pyplot as plt
# plt.imshow(generator(zc).data[0].reshape(28,28),'gray')