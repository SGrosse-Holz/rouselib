"""
(c) Simon Grosse-Holz, 2020

Some code used for force inference in a Rouse model. Given the trajectory of
some polymer locus, infer the force that the polymer exerts on it, assuming
that we start from equilibrium and there are no other forces acting (i.e. we
are dealing with one force pulling at one point of the polymer).

Units
-----
Once calibrated (using MSD data), the code works with physical units.
We use s, μm, pN.
"""

import os, sys

import numpy as np
import scipy.linalg as la
import scipy.special

class locusInference:
    def __init__(self, t, x=None, Gamma=None, tR=None, alpha=0.5, shift=1):
        """
        Initialize for trajectories sampled at times t. The switches in force
        are shifted backward by a fraction s, i.e. the force switches from
        f[i-1] to f[i] at t[i] - s*(t[i] - t[i-1]). Note that this means M is
        singular for s=0.

        NOTE: shifts other than 1 screw with the friction inference. This
        parameter exists mostly for historic reasons.

        alpha is the exponent used in the memory kernel when inferring f from x
        (or the other way round). So far, it is purely heuristically inserted,
        but maybe it's useful. TODO: do the theory for this.

        Input
        -----
        t : (N,) array
            sampling times in seconds
        x : (N, ...) array
            locus trajectory in μm.
            Note that x[0] should be the equilibrium position of the locus
            (i.e. the math assumes that the locus is in equilibrium at the
            start of the trajectory. Everything else wouldn't make sense,
            because we would have to specify the whole polymer conformation.)
            x can have multiple dimensions. The first one is assumed to be
            time.
        Gamma : float
            MSD prefactor, used for calibration. Should be in μm^2/s**alpha
        tR : Rouse time of a finite tether on one side. If None (default), both
            tethers will be infinite. Mathematically this of course simply
            means tR = inf.
            NOTE: So far, a finite tether assumes alpha = 0.5.
        alpha : float in (0, 1)
            Scaling exponent of the MSD, i.e. this controls viscoelasticity of
            the medium. Note that this is incorporated by me tweaking the
            formulas, so no guarantees on correctness for any alpha != 0.5.
            That being said, doesn't look to bad.
        s : float
            relative shift of the force switches
        """
        # if shift != 1:
            # print("Warning: s != 1 screws with the friction inference!")
        if tR is not None and alpha != 0.5:
            print("Warning: finite tether not supported for alpha != 0.5!")

        self.t = t
        self.x = x
        self.tR = tR
        self.alpha = alpha
        self.s = shift

        if Gamma is not None:
            self._calibrate(Gamma)
        else:
            self._kT = 1
            self._spgk = 1
            self.Gamma = self._kT/self._spgk

        self.updateM(tR=tR)

    def _calibrate(self, Gamma, kT=4.114e-3):
        """
        Get prefactors in SI, assuming that
            Γ is in μm^2/s^alpha
            kT is in pN*μm
        """
        self._kT = kT
        self._spgk = kT / Gamma # sqrt(pi * gamma * kappa) in the Rouse model
        self.Gamma = Gamma
        self.updateM()
        
    def updateM(self, tR=None):
        if tR is None:
            def resqrt(x):
                return ((1+np.sign(x))/2*x)**self.alpha
        else:
            def resqrt(x):
                ind = x > 1e-10
                ret = np.zeros_like(x)
                ret[ind] = np.sqrt(x[ind])*(1-np.exp(-np.pi**2*tR/x[ind])) + np.pi**(3/2)*np.sqrt(tR)*scipy.special.erfc(np.pi*np.sqrt(tR/x[ind]))
                return ret

        tforce = list((1-self.s)*self.t[1:] + self.s*self.t[:-1])
        tforce = np.array([self.t[0]-self.t[1]+tforce[0]] + tforce + [self.t[-1] + tforce[-1] - self.t[-2]])
        self.M = 1/self._spgk * ( \
                    resqrt(np.expand_dims(self.t, 1) - \
                           np.expand_dims(tforce[:-1], 0) ) - \
                    resqrt(np.expand_dims(self.t, 1) - \
                           np.expand_dims(tforce[1:], 0) ) )
        self.invM = la.inv(self.M)

### The mathematical basis ###
    def _generate(self, f):
        """
        Input
        -----
        f : force trajectory, in pN

        Output
        ------
        x : locus trajectory, in μm
        """
        assert len(f) == len(self.t)
        return self.M @ f

    def _infer(self, x):
        """
        Basic inference, i.e. for a generic point on the polymer 

        Input
        -----
        x : locus trajectory, in μm

        Output
        ------
        f : force trajectory, in pN
        """
        assert len(x) == len(self.t)
        return self.invM @ (x - np.expand_dims(x[0], 0))
### ###

    def populate(self, x=None, giveOutput=False):
        """
        Main workhorse. Takes self.x and does the full inference, saving
        intermediate results in class attributes.

        This is basically a wrapper for self._infer, just that it also
        calculates some other stuff that could be useful, like velocities.

        Output
        ------
        inferred force, in pN
        """
        if x is not None:
            self.x = x

        # Calculate velocity AT the t[i], where we assume that the locus does
        # not move before and after the experiment
        self.vAt = ( (1-self.s)*(self.x[2:]-self.x[1:-1]) + self.s*(self.x[1:-1]-self.x[:-2]) ) / \
                   ( (1-self.s)*(self.t[2:]-self.t[1:-1]) + self.s*(self.t[1:-1]-self.t[:-2]) )[:, None]
        self.vAt = np.array([(1-self.s)*(self.x[1]-self.x[0])/(self.t[1]-self.t[0])] + \
                            list(self.vAt) + \
                            [self.s*(self.x[-1]-self.x[-2])/(self.t[-1]-self.t[-2])])
        
        # The basic inference
        self.fpoly = -self._infer(self.x)

        # # Remove viscuous ball force
        # self.g = np.sum(np.diff(self.fraw)*np.diff(self.vAt)) / np.sum(np.diff(self.vAt)**2)
        # self.finf = self.fraw - self.g*self.vAt

        if giveOutput:
            return self.fpoly

    def difMagnetic(self, fmagnetic):
        """
        Calculate the difference to the measured magnetic force
        """
        self.fmagnetic = fmagnetic
        self.funex = -self.fmagnetic - self.fpoly

### Noise ###
    def covTrajectory(self):
        """
        Covariance matrix for the trajectory, for fixed x[0].  This is given by

            S(t, t') = 1/2*(MSD(t) + MSD(t') - MSD(|Δt|)) .
            
        Use .util.sampleCov0() to sample from this.
        """
        t0 = np.expand_dims(self.t, 0) - self.t[0]
        t1 = np.expand_dims(self.t, 1) - self.t[0]

        return 0.5*self.Gamma*( t0**self.alpha + t1**self.alpha - np.abs(t0-t1)**self.alpha )

    def covForce(self):
        """
        Covariance matrix for the force, given the covariance matrix of the
        trajectory. This is simple Gaussian error propagation:

            S_force = M^{-1} @ S @ M^{-T} .

        Use .util.sampleCov0() to sample from this.
        """
        return self.invM @ self.covTrajectory() @ self.invM.T

### Dragging ###
    def computeFdrag(self, density, mode=0, ix=1, giveOutput=False, giveTrajs=False):
        """
        Calculate the additional force exerted on the locus, if it has to drag
        local density with it.

        Input
        -----
        density : (N,) array
            the local density at the position of the locus. This is basically
            just a local proportionality factor. As such, it will be included
            linearly (there was an idea at some point that dependency on the
            density should be a square root... who knows).
        mode : integer
            specifies the model to use. So far implemented:
              0 = sticky chromatin
              1 = two-sided glove
              2 = one-sided glove
        ix : component of x for which to calculate.

        Output
        ------
        fdrag : the force exerted by all the dragged stuff
        """
        if len(self.x.shape) == 1:
            x = self.x
            v = self.vAt
        else:
            x = self.x[:, ix]
            v = self.vAt[:, ix]

        if giveTrajs:
            trajs = np.zeros((len(x), len(x)))
        fdrag = np.zeros_like(x)
        for j in range(len(x)):
            offset = x[j]
            traj = x - offset
            traj[:j] = 0

            moveForward = v[j] > 0 # If moving backwards, the glove also works backwards!
            if mode == 2:
                moveForward = True

            while True:
                F = self._infer(traj)
                ind = np.where(F < -1e-10 if moveForward else F > 1e-10)[0]

                if len(ind) == 0 or mode == 0:
                    break
                ind = ind[0]

                F[ind:] = 0
                traj = self._generate(F)
                traj[ind:] = (np.maximum if moveForward else np.minimum)(traj[ind:], x[ind:]-offset)

            fdrag += density[j]*F
            if giveTrajs:
                trajs[:, j] = traj

        # Handle all the output possibilities
        if len(self.x.shape) == 1:
            self.fdrag = fdrag
        else:
            if not hasattr(self, 'fdrag'):
                self.fdrag = np.empty(self.x.shape)
                self.fdrag[:] = np.nan
            self.fdrag[:, ix] = fdrag

        if giveOutput and giveTrajs:
            return fdrag, trajs
        elif giveOutput:
            return fdrag
        elif giveTrajs:
            return trajs
