import os, sys

import numpy as np
import scipy.linalg

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
