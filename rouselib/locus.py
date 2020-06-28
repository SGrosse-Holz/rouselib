"""
Some useful functions for dealing with locus pulling. Specifically, generating
distance trajectories from force trajectories and inferring vice versa. This is
a linear problem, so the main point of this module is to code up generation of
the matrix M in 

    x_i = M_ij @ f_j

where x_i is x(t_i) and f_i is the force acting on the locus between times
t_{i-1} and t_i, where t_0 = 0.

Convention used here: t, x, f should always be starting with the corresponding
0th entry, which will be set to 0 explicitly. For t and x, this results in a
shift by -t[0] or -x[0]. For f, we will just ignore f[0].
"""

import os,sys

import numpy as np
import scipy.linalg

def M(t):
    """
    Generate M from the time points t. Explicitly:

        M_ij = ReSqrt(t_i - t_{j-1}) - ReSqrt(t_i - t_j) ,

    where ReSqrt(x) is 0 for x < 0, Sqrt(x) for x >= 0.

    Input
    -----
    t : list of times at which the trajectories are evaluated.

    Notes
    -----
    We choose the convention that M should be (len(t), len(t)). This means
    introducing t_{-1} = t_0, which makes the matrix singular. This fits the
    usual pattern in this module, where the [0] entry of trajectories is always
    fixed to 0 and the invertible part of matrices is M[1:, 1:].
    """
    def resqrt(x):
        return np.sqrt(np.abs(x))*(1+np.sign(x))/2

    t = np.insert(t - t[0], 0, 0)
    return resqrt(np.expand_dims(t[1:], 1) - \
                  np.expand_dims(t[:-1], 0) ) - \
           resqrt(np.expand_dims(t[1:], 1) - \
                  np.expand_dims(t[1:], 0) )

def generate(t, f):
    """
    Generate a Rouse trajectory for a locus pulled with f_i for t_{i-1} < t <=
    t_i

    Input
    -----
    t : list of time points
    f : list of forces (f[0] will be ignored)

    Output
    ------
    x : list of positions
    """
    t = t - t[0]
    return M(t) @ f / np.sqrt(np.pi)

def infer(t, x):
    """
    Infer the force f from a locus trajectory x. Remember that this assumes
    that t_0 = 0, x_0 = 0 but the first entries in 

    Input
    -----
    t : list of time points
    x : list of positions

    Output
    ------
    f : list of forces
    """
    t = t - t[0]
    x = x - x[0]
    return np.insert(scipy.linalg.inv(M(t)[1:, 1:]) @ x[1:] * np.sqrt(np.pi), 0, 0)

def Scov(t):
    """
    Calculate the covariance matrix for a tracked monomer sampled at time
    points t. This is given by

        S(t, t') = sqrt(t-t[0]) + sqrt(t'-t[0]) - sqrt(abs(t-t')) .

    To get the units right, multiply with the MSD prefactor Gamma.
    """
    t0 = np.expand_dims(t, 0) - t[0]
    t1 = np.expand_dims(t, 1) - t[0]

    return np.sqrt(t0) + np.sqrt(t1) - np.sqrt(np.abs(t1-t0))

def unscaled_polymer_noise(t, d=1):
    """
    Give samples of the covariance a Rouse polymer would have at times t,
    assuming that the trajectory is fixed to some constant (probably 0) at
    t[0].

    To get the units right, multiply with sqrt of the MSD prefactor Gamma.
    """
    return np.insert(scipy.linalg.cholesky(Scov(t)[1:, 1:], lower=True) @ np.random.normal(size=(len(t)-1, d)), 0, 0)

def inference_cov(t):
    """
    Covariance matrix for the force inference, assuming the trajectory to have
    Rouse covariance. Note that since we don't know the temperature, there is a
    relative factor between these that we don't know.

    This covariance is given by

        C_inf = pi * inv(M(t)) @ Scov(t) @ inv(M(t)).T ,

    owing to the fact that the trajectory has covariance Scov.
    """
    invM = scipy.linalg.inv(M(t)[1:, 1:])
    ret = np.zeros((len(t), len(t)))
    ret[1:, 1:] = np.pi * invM @ Scov(t)[1:, 1:] @ invM.T
    return ret

def unscaled_force_noise(t):
    """
    Draw from the distribution given by inference_cov.
    """
    return np.insert(scipy.linalg.cholesky(inference_cov(t)[1:, 1:], lower=True) @ np.random.normal(size=(len(t)-1,)), 0, 0)
