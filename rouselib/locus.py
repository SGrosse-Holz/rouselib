"""
Some useful functions for dealing with locus pulling. Specifically, generating
distance trajectories from force trajectories and inferring vice versa. This is
a linear problem, so the main point of this module is to code up generation of
the matrix M in 

    x_i = M_ij @ f_j

where x_i is x(t_i) and f_i is the force acting on the locus between times
t_{i-1} and t_i, where t_0 = 0.
"""

import os,sys

import numpy as np
import scipy.linalg

def M(t):
    """
    Generate M from the time points t_i. Explicitly:

        M_ij = ReSqrt(t_i - t_{j-1}) - ReSqrt(t_i - t_j) ,

    where ReSqrt(x) is 0 for x < 0, Sqrt(x) for x >= 0.

    Input
    -----
    t : list of times at which the trajectories are evaluated. This should
        start with t_1 > 0, since t_0 = 0 will be added internally.
    """
    def resqrt(x):
        return np.sqrt(np.abs(x))*(1+np.sign(x))/2

    t_i = t
    ti1 = np.array([0]+list(t_i[:-1])) # t_{i-1} with t_0 = 0

    return resqrt(np.expand_dims(t_i, 1) - \
                  np.expand_dims(ti1, 0) ) - \
           resqrt(np.expand_dims(t_i, 1) - \
                  np.expand_dims(t_i, 0) )

def generate(t, f):
    """
    Generate a Rouse trajectory for a locus pulled with f_i for t_{i-1} < t <=
    t_i

    Input
    -----
    t : list of time points
    f : list of forces

    Output
    ------
    x : list of positions
    """
    return M(t) @ f

def infer(t, x):
    """
    Infer the force f from a locus trajectory x

    Input
    -----
    t : list of time points
    x : list of positions

    Output
    ------
    f : list of forces
    """
    return scipy.linalg.inv(M(t)) @ x
