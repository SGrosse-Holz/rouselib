import os, sys

import numpy as np
import scipy.linalg

from . import util

class rousesim:
    """
    A class for simulation / exact solution of a Rouse model with eom
    $$
    \dot{x} = -kBx + F + sigma\eta\,.
    $$

    Note that while k, B, F, sigma are publicly accessible, setup() should be
    called after changing them.
    """

    def __init__(self, N, k=1, sigma=1):
        """
        Set up a new simulation of a simple chain of length N, tethered to the
        origin. Also set spring constant k and noise level sigma.
        """
        self.N = N
        self.k = k
        self.sigma = sigma

    def set_BFfree(self):
        """
        Set B and F to the values for a free chain. Note that this will not run
        as simulation, because B is singular.
        """
        self.B = util.Bfree(self.N)
        self.F = np.zeros((self.N, 3))

    def add_crosslinks(self, links):
        self.B += util.Bcrosslinks(self.N, links)

    def add_tether(self, pos=0, strength=1):
        """Add additional tether(s)"""
        self.B[pos, pos] += strength

    def _s2_2k(self):
        return self.sigma**2 / (2*self.k)

    def setup(self, dt):
        """
        Set up this instance for simulation at time step dt.
        """
        self._dt = dt
        self._invB = scipy.linalg.inv(self.B)
        self._A = scipy.linalg.expm(-self.k*self._dt*self.B)

        # Mean
        self.update_G()

        # Variance
        self._Sig = (np.eye(self.N) - self._A@self._A) @ self._invB * self._s2_2k()
        self._LSig = scipy.linalg.cholesky(self._Sig, lower=True)

    def update_G(self):
        self._G = (np.eye(self.N) - self._A) @ (self._invB/self.k) @ self.F

    ###### Actually running a simulation

    def conf_ss(self):
        """
        Draw a conformation from steady state. This function does not
        necessarily require setup() having been called.

        Note: do not fiddle with _G's here, they have nothing to do with the
        steady state.
        """
        if not hasattr(self, '_invB'):
            self._invB = scipy.linalg.inv(self.B)

        L = scipy.linalg.cholesky(self._invB * self._s2_2k(), lower=True)
        return L @ np.random.normal(size=(self.N, 3)) + (self._invB/self.k) @ self.F

    def propagate(self, conf, deterministic=False):
        """
        Propagate a given conformation by the time step given to setup().
        """
        if deterministic:
            return self._A @ conf + self._G
        else:
            return self._A @ conf + self._G + self._LSig @ np.random.normal(size=(self.N, 3))

    def propagate_gen(self, conf, steps, **kwargs):
        for _ in range(steps):
            conf = self.propagate(conf, **kwargs)
            yield conf

    ######## Stuff to do with units

    def calibrate(self, dmon_nm=np.sqrt(1000), Gamma_um2_s05=0.01, kBT_pNnm=4, print_gamma=False):
        """
        Calculate length and time scales for the simulation
        """
        self.units = { \
                't_s' : 4*(dmon_nm/1e3)**4*self.k/(np.pi*Gamma_um2_s05**2), \
                'x_nm' : dmon_nm * np.sqrt(2*self.k/(3*self.sigma**2)), \
                'F_pN' : np.sqrt(3/(2*self.k))/self.sigma * kBT_pNnm / dmon_nm }

        if print_gamma:
            print('damping coefficient for one monomer: gamma = {:e} kg/s'.format( \
                    24*dmon_nm**2/(np.pi*Gamma_um2_s05**2)*1e-15))

    def print_units(self):
        if not hasattr(self, 'units'):
            raise ValueError('self.units undefined, run calibrate first')

        print("Simulation units:")
        print("time   : {:.2e} s".format(self.units['t_s']))
        print("length : {:.2e} nm".format(self.units['x_nm']))
        print("force  : {:.2e} pN".format(self.units['F_pN']))

    def Rouse_time(self, do_print=False):
        tR = (self.N / np.pi)**2 / self.k
        if do_print:
            print("Rouse time: {:.2e} simulation time units".format(tR))
        else:
            return tR

    def Equilibration_time(self, do_print=False):
        """
        This is defined as the crossover time in the MSD. It has a factor of
        pi**3/4 relative to the Rouse time
        """
        tR = np.pi * self.N**2 / (4*self.k)
        if do_print:
            print("Equilibration time: {:.2e} simulation time units".format(tR))
        else:
            return tR
