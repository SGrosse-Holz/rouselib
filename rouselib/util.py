import os, sys

import numpy as np
import scipy.linalg

import matplotlib.pyplot as plt

def Bfree(N):
    """
    Generate a NxN Rouse connectivity matrix with free ends.
    """

    B = 2*np.identity(N)
    B[0, 0] = B[-1, -1] = 1
    for i in range(1, N):
        B[i, i-1] = B[i-1, i] = -1
    return B

def Bcrosslinks(N, links):
    """
    Generate connectivity matrix for crosslinks, and only the crosslinks. So a
    reasonable full matrix would be Bfree(N) + Bcrosslinks(N, ...)

    Input
    -----
    N : int
        length of the chain
    links : list of 2-tuple
        crosslinks to add
    """
    links = np.array(links, dtype=int)
    B = np.zeros((N, N))
    for link in links:
        B[link[0], link[0]] += 1
        B[link[0], link[1]] -= 1
        B[link[1], link[0]] -= 1
        B[link[1], link[1]] += 1
    return B

def optimalRescaling(trajs):
    """
    Find the optimal scale factors to make the trajectories as similar as
    possible. This is useful if we want to look at the mean behavior of
    trajectories that we expect to have similar behaviors when rescaled
    appropriately.

    Input
    -----
    trajs : (N, T) array
        array of N trajectories, each of length T

    Output
    ------
    w : (N,) array
        optimal weights, sum(w) = 1
    """
    N = trajs.shape[0]
    FF = np.tensordot(trajs, trajs, (1, 1))
    NFF = FF - N*np.eye(N)*FF
    w = scipy.linalg.inv(NFF) @ np.ones(N)
    return w / np.sum(w)

def sampleCov(S, n=1):
    """
    Draw n zero mean Gaussian samples with covariance matrix S, using Cholesky
    decomposition.
    """
    L = scipy.linalg.cholesky(S, lower=True)
    if n > 1:
        return L @ np.random.randn(len(S), n)
    else:
        return L @ np.random.randn(len(S))

def sampleCov0(S, n=1):
    """
    Draw samples from a covariance matrix whose first entry on the diagonal is
    0. That dimension is omitted and a zero manually prepended to the sample.
    """
    return np.insert(sampleCov(S[1:, 1:], n), 0, 0, axis=0)
