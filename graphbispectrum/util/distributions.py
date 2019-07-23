#!/usr/bin/env python
# encoding: utf-8
"""
distributions.py

Created by Ram Mehta on 2009-11-18.
Copyright (c) 2009 . All rights reserved.
Updated: Kilian Koepsell and Chris Hillar 2010-2011
"""

import numpy as np
import unittest

class Sampler(object):
    """ Sampler object

    Contains dataset, (M x N) - dim. *numpy* array,
    M samples of dimension N
    
    Sampler() Returns a single sample, when called without argument.
    Returns s samples (as an s x n numpy array), when called with integer argument s.

    Parameters
    ----------
    datatset = input *numpy* array containing data to sample (otherwise it will make it one)
    sample = function that returns a sample
    """

    def __init__(self, dataset=[], sample=None, use_gpu=False, dtype=np.float32):
        self.use_gpu = use_gpu
        self.dtype = dtype
        dataset = np.asarray(dataset).astype(self.dtype)
        dataset = np.atleast_2d(dataset)
        m,n = dataset.shape
        if use_gpu:
            self.dataset = ga.to_gpu(dataset)
            self.sample_gpu = ElementwiseKernel(
                "float *out, float *dataset, int dim, int *idx",
                "out[i] = dataset[idx[i/dim]+i%dim]",
                "sample_on_gpu")
        else:
            self.dataset = dataset
        self.m = m
        self.n = n
        # when sample function not supplied uniform over data is assumed
        if sample is not None:
            self.sample = sample
        else:
            self.sample = None

    def _sample(self, s=None): 
        # uniform over dataset is sampler if none specified in construction
        if self.use_gpu:            
            idx = ga.to_gpu(np.random.randint(self.m, size=s))
            out = ga.empty((s, self.n), dtype=self.dtype)
            self.sample_gpu(out, self.dataset, self.n, idx)
            return out
        else:
            return self.dataset[np.random.randint(self.m, size=s)]

    def __call__(self, s=None):
        if self.sample is None:
            return self._sample(s)  # uniform over data if no sample function supplied
        else:
            return self.sample(s)


class Factory:
    """ Produces samplers.
    """

    def __init__(self, n):
        self.n = n

    def uniform_distribution_sampler(self, m, use_gpu=False):
        """
        returns a sampler uniform over m fixed n-length np.random.rand vectors 
        """
        return Sampler(np.random.rand(m, self.n))

    def uniform_from_dataset_sampler(self, dataset, use_gpu=False):
        """
        returns a sampler which draws uniformly dataset vectors when sampling
        """
        return Sampler(dataset, use_gpu=use_gpu)

    def normal_distribution_sampler(self, m, use_gpu=False):
        """
        create m samples of a normal distribution
        return a sampler which samples uniformly over these m samples
        """
        return Sampler(np.random.randn(m, self.n), use_gpu=use_gpu)

    def bernoulli_distribution_binary_sampler(self, m, sparsity=None, use_gpu=False):
        """
        create m samples of a k-sparse bernoulli distribution
        return a sampler which samples uniformly over these m samples
        """
        if sparsity is None:
            data = np.random.randint(2, size=(m, self.n))
        else:
            data = np.zeros((m, self.n), dtype=int)
            idx = (np.random.rand(m, self.n).argsort(1)[:,:sparsity]+self.n*np.arange(m)[:,None]).ravel()
            data.ravel()[idx] = 1
        return Sampler(data, use_gpu=use_gpu)

    def bernoulli_distribution_vector_binary_sampler(self, m, v, use_gpu=False):
        """
        create a sampler having m binary vectors b of length n
        with b(i) being 1 having probability v(i)
        """
        rand_vect = np.random.rand(m, self.n)
        outcome = v > rand_vect
        data = outcome.astype('int')
        return Sampler(data, use_gpu=use_gpu)


    def bernoulli_distribution_corrupted_binary_sampler(self, m, sparsity=None, avg_bits_corrupted=None, bits_corrupted=None, use_gpu=False):
        """
        create k-sparse bernoulli distribution
        """
        sampler = self.bernoulli_distribution_binary_sampler(m, sparsity=sparsity, use_gpu=use_gpu)
        return corrupt_binary_sampler(sampler, avg_bits_corrupted=avg_bits_corrupted, bits_corrupted=bits_corrupted)

    def bernoulli_distribution_bipolar_sampler(self, m, sparsity=None, use_gpu=False):
        if sparsity is None:
            data = 2*np.random.randint(2, size=(m, self.n)) - 1
        else:
            data = -np.ones((m, self.n), dtype=int)
            idx = (np.random.rand(m, self.n).argsort(1)[:,:sparsity]+self.n*np.arange(m)[:,None]).ravel()
            data.ravel()[idx] = 1
        return Sampler(data, use_gpu=use_gpu)


def corrupt_binary_data(data, avg_bits_corrupted=None, bits_corrupted=None):
    ndim = data.ndim
    data = np.atleast_2d(data)
    size, n = data.shape

    if avg_bits_corrupted is not None:
        p = 1.0*avg_bits_corrupted/n  # prob bit flipped
        which_to_flip = np.random.rand(size,n) < p
    elif bits_corrupted is not None:
        which_to_flip = np.random.rand(size, n).argsort(1) < bits_corrupted
    else:
        raise ValueError('either bits_corrupted or avg_bits_corrupted option has to be given')

    out = data.copy()
    out[which_to_flip] *= -1
    out[which_to_flip] += 1

    if ndim == 1: return out[0]
    return out

def corrupt_binary_sampler(sampler, avg_bits_corrupted=None, bits_corrupted=None):
    def sample_corrupted(s=None):
        return corrupt_binary_data(sampler(s), avg_bits_corrupted=avg_bits_corrupted, bits_corrupted=bits_corrupted)
    
    return Sampler(sampler.dataset, sample=sample_corrupted)


class FactoryTests(unittest.TestCase):
    def setUp(self):
        self.factory = Factory(20)
        self.use_gpu = False

    def test_uniform_distribution_sampler(self):
        sampler = self.factory.uniform_distribution_sampler(5, use_gpu=self.use_gpu)
        dataset = sampler.dataset
        if type(dataset) ==  ga.GPUArray: dataset = dataset.get()
        for i in range(0, 100):
            assert(sampler() in dataset)

    def test_sparse_bernoulli_distribution_sampler(self):
        sampler = self.factory.bernoulli_distribution_binary_sampler(10, sparsity=3, use_gpu=self.use_gpu)
        for i in range(0, 100):
            assert(sampler().sum() == 3)

    def test_corrupt_data(self):
        sampler = self.factory.bernoulli_distribution_binary_sampler(10**4, sparsity=3, use_gpu=self.use_gpu)
        dataset = sampler.dataset
        if type(dataset) ==  ga.GPUArray: dataset = dataset.get()
        data_corrupted = corrupt_binary_data(dataset, 2)
        np.testing.assert_array_almost_equal(np.abs(dataset-data_corrupted).sum(1).mean(), 2, decimal=1)


if __name__ == '__main__':
    print("Test for this file not implemeneted yet!")

