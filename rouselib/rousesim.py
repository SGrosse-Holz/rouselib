import os, sys

import numpy as np
import scipy.linalg
import scipy.integrate

from . import util

class Rousesim:
    """
    A class for simulation / exact solution of a Rouse model with eom
    $$
    \dot{x} = -k*A*x + F + sigma\eta\,.
    $$

    Note that while k, A, F, sigma are publicly accessible, setup() should be
    called after changing them.

    This class also has a parameter 'deterministic'. If set to True, this will
    remove all thermal noise, such that the configuration evolves as the
    ensemble mean. This is interesting when studying the effect of forces on
    the chain.
    """

    def __init__(self, N, k=1, sigma=1, deterministic=False, d=3):
        """
        Set up a new simulation of a simple chain of length N, tethered to the
        origin. Also set spring constant k and noise level sigma.
        """
        self.N = N
        self.k = k
        self.sigma = sigma
        self.deterministic = deterministic
        self.d = d

    def set_AFfree(self):
        """
        Set A and F to the values for a free chain. Note that this will not run
        as simulation, because A is singular.
        """
        self.A = util.Afree(self.N)
        self.F = np.zeros((self.N, self.d))

    def add_crosslinks(self, links, relStrength=1):
        """
        Add crosslinks to the connectivity matrix. It's strength relative to
        the backbone bonds can be adapted with relStrength.
        """
        self.A += relStrength * util.Acrosslinks(self.N, links)

    def add_tether(self, pos=0, relStrength=1, to=None):
        """Add additional tether(s)"""
        if to is None:
            to = np.zeros(self.d)
        self.A[pos, pos] += relStrength * self.k
        self.F[pos] += relStrength * self.k * to

    def _s2_2k(self):
        return self.sigma**2 / (2*self.k)

    def setup(self, dt):
        """
        Set up this instance for simulation at time step dt.
        """
        self._dt = dt
        if np.isclose(scipy.linalg.det(self.A), 0):
            self._invA = None
        else:
            self._invA = scipy.linalg.inv(self.A)
        self._B = scipy.linalg.expm(-self.k*self._dt*self.A)

        # Mean
        self.update_G()

        # Variance
        self.update_Sig()

    def update_G(self):
        """ Update G from k, A, F, sigma """
        if not np.any(self.F):
            self._G = np.zeros_like(self.F)
        elif self._invA is not None:
            self._G = (np.eye(self.N) - self._B) @ (self._invA/self.k) @ self.F
        else:
            def integrand(tau):
                return scipy.linalg.expm(-self.k*self.A*tau) @ self.F
            self._G = scipy.integrate.quad_vec(integrand, 0, self._dt)[0]

    def update_Sig(self):
        """ Update Sigma and its Cholesky decomposition from k, A, F, sigma """
        if self._invA is not None:
            self._Sig = (np.eye(self.N) - self._B@self._B) @ self._invA * self._s2_2k()
        else:
            def integrand(tau):
                eAt = scipy.linalg.expm(-self.k*self.A*tau)
                return eAt @ eAt.T * self.sigma**2
            self._Sig = scipy.integrate.quad_vec(integrand, 0, self._dt)[0]

        self._LSig = scipy.linalg.cholesky(self._Sig, lower=True)

    ###### Actually running a simulation

    def conf_ss(self):
        """
        Draw a conformation from steady state. This function does not
        necessarily require setup() having been called.

        Note: do not fiddle with _G's here, they have nothing to do with the
        steady state.
        """
        if not hasattr(self, '_invA') or self._invA is None:
            try:
                self._invA = scipy.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                raise RuntimeError("A is singular, there is no steady state to sample from.")

        conf = (self._invA/self.k) @ self.F
        if not self.deterministic:
            L = scipy.linalg.cholesky(self._invA * self._s2_2k(), lower=True)
            conf += L @ np.random.normal(size=(self.N, self.d))
        return conf

    def propagate(self, conf):
        """
        Propagate a given conformation by the time step given to setup().
        """
        if self.deterministic:
            return self._B @ conf + self._G
        else:
            return self._B @ conf + self._G + self._LSig @ np.random.normal(size=(self.N, self.d))

    def propagate_gen(self, conf, steps, **kwargs):
        for _ in range(steps):
            conf = self.propagate(conf, **kwargs)
            yield conf

    def contact_probability(self):
        J = self._s2_2k() * self._invA
        Jii = np.tile(np.diagonal(J), (len(J), 1))
        return (Jii + Jii.T - 2*J)**(-3/2)

    def traj_noise_acf(self, m, T):
        """
        Calculate the noise autocorrelation ('centered acf') of the trajectory
        (m.x)(t), for T steps. The covariance matrix for this trajectory is
        scipy.linalg.toeplitz(acf).

        Input
        -----
        m : (N,) array-like
            the measurement vector
        T : int
            number of steps (i.e. dimension of the covariance matrix)
        """
        Jm = self._invA * self._s2_2k() @ m
        return np.array([m @ np.linalg.matrix_power(self._B, n) @ Jm for n in range(T)])

    ######## MSD

    def analytical_MSD(self, dts, w=None):
        """
        Calculate the analytical MSD ``w @ <( x(t+Δt) - x(t) )^2> @ w``.

        Parameters
        ----------
        dt : array-like
            vector of time lags for which to evaluate the MSD
        w : (N,) array_like, optional
            the measurement vector. E.g. ``[-1, 0, 0, ..., 0, 0, 1]`` if we are
            interested in end-to-end distance. If unspecified, return full
            covariance matrices.

        Returns
        -------
        np.ndarray
            MSD evaluated at the given time lags

        Notes
        -----
        For the discrete process ``x_n = B.x_{n-1} + ξ_n`` with ``<ξ_m x ξ_n> =
        δ_mn Σ`` we have ``<( x_n - x_{n-1} )^(x)2> = Σ + (1+B)^-1 Σ (1-B)``.

        We assume the customary ``BΣ = ΣB^T``.
        """
        dts = np.sort(dts)

        if not hasattr(self, '_invA'):
            if np.isclose(scipy.linalg.det(self.A), 0):
                self._invA = None
            else:
                self._invA = scipy.linalg.inv(self.A)

        Bs = [scipy.linalg.expm(-self.k*self.A*dt) for dt in dts]

        if self._invA is not None:
            Sigs = [(np.eye(self.N) - B @ B) @ self._invA * self._s2_2k() for B in Bs]
        else:
            def integrand(tau):
                eAt = scipy.linalg.expm(-self.k*self.A*tau)
                return eAt @ eAt.T * self.sigma**2
            # Some trickery to calculate cum_quad_vec
            res, err, full_info = scipy.integrate.quad_vec(integrand, 0, np.max(dts), full_output=True, points=dts)
            Sigs = [np.sum(full_info.integrals[full_info.intervals[:, 1] <= dt], axis=0) for dt in dts]

        xxs = [2*scipy.linalg.inv(np.eye(self.N) + B) @ Sig for B, Sig in zip(Bs, Sigs)]

        if w is None:
            return np.array(xxs)
        else:
            return np.array([w @ xx @ w for xx in xxs])


    ######## Stuff to do with units

    def calibrate(self, dmon_nm=np.sqrt(1000), Gamma_um2_s05=0.01, kBT_pNnm=4, print_gamma=False):
        """
        Calculate length and time scales for the simulation
        """
        self.units = { \
                't_s' : 4*(dmon_nm/1e3.flatten().flatten())**4*self.k/(np.pi*Gamma_um2_s05**2), \
                'x_nm' : dmon_nm * np.sqrt(2*self.k/(self.d*self.sigma**2)), \
                'F_pN' : np.sqrt(self.d/(2*self.k))/self.sigma * kBT_pNnm / dmon_nm }

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
        pi**3 / 4 relative to the Rouse time
        """
        tR = np.pi * self.N**2 / (4*self.k)
        if do_print:
            print("Equilibration time: {:.2e} simulation time units".format(tR))
        else:
            return tR
