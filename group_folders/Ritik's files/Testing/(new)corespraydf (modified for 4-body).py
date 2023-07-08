""" The corespraydf class

"""

__author__ = "Steffani Grondin & Jeremy J Webb"

__all__ = [
    "corespraydf",
]

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, PlummerPotential, KingPotential, \
    MovingObjectPotential
from galpy.util import conversion, coords
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

import time


class corespraydf(object):
    """ A class for initializing a distribution function for stars that are
        ejected from the core of a globular cluster
    -- If emin and emax are None, assume limits are between twice the hard-soft
       boundary and twice the contact boundary between two solar mass stars
    Parameters
    ----------

    gcorbit : string or galpy orbit instance
        Name of Galactic Globular Cluster from which to simulate core ejection
        or a Galpy orbit instance
    pot : galpy potential
        Potential to be used for orbit integration (default: MWPotential2014)
    mgc : float
        globular cluster mass - needed if cluster's potential is to be included
        in orbit integration of escapers (default: None)
    rgc : float
        half-mass radius of globular cluster (assuming Plummer potential) or
        tidal radius of globular cluster (assuming King potential) (default: None)
    W0 : float
        King central potential parameter (default: None, which results in
        cluster potential taken to be a Plummer)
    ro : float
        galpy length scaling parameter (default: 8.)
    vo : float
        galpy velocity scaling parameter (default: 220.)


    History
    -------
    2021 - Written - Grondin (UofT)

    """

    def __init__(self, gcorbit, pot=MWPotential2014, mgc=None, rgc=None,
                 W0=None, ro=8., vo=220., verbose=False, timing=False):

        if isinstance(gcorbit, str):
            self.gcname = gcorbit
            self.o = Orbit.from_name(self.gcname, ro=ro, vo=vo,
                                     solarmotion=[-11.1, 24.0, 7.25])
        else:
            self.gcname = 'unknown'
            self.o = gcorbit

        self.ro, self.vo = ro, vo
        self.to = conversion.time_in_Gyr(ro=self.ro, vo=self.vo) * 1000.
        self.mo = conversion.mass_in_msol(ro=self.ro, vo=self.vo)

        self.mwpot = pot

        if mgc is None:
            self.gcpot = None
        else:
            if W0 is None:
                ra = rgc / 1.3
                self.gcpot = PlummerPotential(mgc / self.mo, ra / self.ro,
                                              ro=self.ro, vo=self.vo)
            else:
                self.gcpot = KingPotential(W0, mgc / self.mo, rgc / self.ro,
                                           ro=self.ro, vo=self.vo)

        self.binaries = False

        self.timing = timing

# -----------------------------------------------------------------------------
    def sample_three_body(self, tdisrupt=1000., rate=1., nstar=None, mu0=0.,
                          sig0=10.0, vesc0=10.0, rho0=1., mmin=0.1, mmax=1.4,
                          alpha=-1.35, masses=None, m1=None, m2a=None, m2b=None,
                          emin=None, emax=None, balpha=-1, q=-3, npeak=5.,
                          binaries=False, verbose=False, **kwargs):
        """ A function for sampling the three-body interaction core ejection
            distribution function

        Parameters
        ----------

        tdisrupt : float
            time over which sampling begins (Myr)
        rate : float
            ejection rate (default 1 per Myr)
        nstar : float
            if set, nstar stars will be ejected randomly from tdisrupt to 0 Myr.
            Rate is recalculated. (default : None)
        mu0 : float
            average 1D velocity in the core (default: 0 km/s)
        sig0 : float
            avarege 1D velocity dispersions in the core (default 10.0 km/s)
        vesc0 : float
            escape velocity from the core (default: 10.0 km/s)
        rho0 : float
            core density (default: 1 Msun/pc^3)
        mgc : float
            globular cluster mass in solar masses - needed if cluster's
            potential is to be included in orbit integration of escapers
            (default: None)
        rgc : float
            half-mass radius of globular cluster (assuming Plummer potential) or
            tidal radius of globular cluster (assuming King potential) in kpc
            (default: None)
        W0 : float
            King central potential parameter (default: None, which results in
            cluster potential taken to be a Plummer)
        mmin : float
            minimum stellar mass in core (default (0.1 Msun))
        mmax : float
            maximum stellar mass in the core (default: 1.4 Msun)
        alpha : float
            slope of the stellar mass function in the core (default: -1.35)
        masses : float
            array of masses to be used instead of drawing for a power-law mass
            function (default: None)
            Note : mmin, mmax, and alpha will be overridden
        m1a : float
            fixed mass for binary star A1 (default: None)
            Note : (mmin, mmax, alpha) or (masses) must still be provided to
            determine the mean mass in the core
        m1b: float
            fixed mass for binary star B1 in first binary (default: None)
        m2a : float
            fixed mass for binary star A2 (default: None)
            Note : (mmin, mmax, alpha) or (masses) must still be provided to
            determine the mean mass in the core
        m2b : float
            fixed mass for binary star B2 (default: None)
            Note : (mmin, mmax, alpha) or (masses) must still be provided to
            determine the mean mass in the core
        emin : float
            minimum binary energy in Joules (default: None)
        emax : float
            maximum binary energy in Joules (default: None)
        balpha : float
            power-law slope of initial binary binding energy distribution
            (default: -1)
        q: float
            exponent for calculating probability of stellar escape from
            three-body system (#Equation 7.23) (default: -3)
        npeak : float
            when sampling kick velocity distribution function, sampling range
            will be from 0 to npeak*vpeak, where vpeak is the peak in the
            distribution function (default: 5)
        binaries : bool
            keep track of binaries that receive recoil kicks greater than the
            cluster's escape velocity (default : False)
            if true, keep track of second binary
            if false, keep track of first binary
            might be arbitrary
        verbose : bool
            print additional information to screen (default: False)

        Key Word Arguments
        ----------
        nrandom : int
            Nunber of random numbers to sample in a given batch
        ntsteps : int
            Number of time steps to take for orbit integration
        rsample : bool
            Sample separation between single star and binary within core
            (default: False)
        nrsep : float
            Numer of mean separations to sample out to when sampling separation
            between single and binary stars (default : 2)
        initialize : bool
            initialize orbits only, do not integrate (default:False)
            Note if initialize == True then the initial, un-integrated orbits,
            will be returned

        Returns
        ----------
        of : orbit
            galpy orbit instance for kicked stars

        if binaries:
            obf : orbit
                galpy orbit instance for recoil binary stars

        History
        -------
        2021 - Written - Grondin/Webb (UofT)

        """

        grav = 4.302e-3  # pc/Msun (km/s)^2
        msolar = 1.9891e30

        nrandom = kwargs.get('nrandom', 1000)
        self.timing = kwargs.get('timing', self.timing)

        self.tdisrupt = tdisrupt

        rsample = kwargs.get('rsample', False)
        initialize = kwargs.get('initialize', False)

        # Select escape times
        # If nstar is not None, randomly select escapers between tstart and tend
        if nstar is not None:
            self.nstar = nstar
            self.rate = nstar / self.tdisrupt
        else:
            self.rate = rate
            self.nstar = self.tdisrupt * rate

        self.tesc = -1. * self.tdisrupt * np.random.rand(self.nstar)

        ntstep = kwargs.get('ntstep', 10000)
        nrsep = kwargs.get('nrsep', 1)
        method = kwargs.get('method', None)

        ts = np.linspace(0., -1. * self.tdisrupt / self.to, ntstep)

        if method is not None:
            self.o.integrate(ts, self.mwpot, method=method)
        else:
            self.o.integrate(ts, self.mwpot)

        if self.gcpot is None:
            self.pot = self.mwpot
        else:
            moving_pot = MovingObjectPotential(self.o, self.gcpot, ro=self.ro,
                                               vo=self.vo)
            self.pot = [self.mwpot, moving_pot]

        self.mu0, self.sig0, self.vesc0, self.rho0 = mu0, sig0, vesc0, rho0

# change the below (want to get
        if masses is None:
            self.mmin, self.mmax, self.alpha = mmin, mmax, alpha
            # Mean separation of star's in the core equal to twice the radius of a sphere that contains one star
            # Assume all stars in the core have mass equal to the mean mass
            self.masses = self._power_law_distribution_function(1000,
                                                                self.alpha,
                                                                self.mmin,
                                                                self.mmax)
        else:
            self.masses = masses

        self.mbar = np.mean(self.masses)
        self.rsep = ((self.mbar / self.rho0) / (4. * np.pi / 3.)) ** (1. / 3.)

        # Limits of binary energy distribution
        # If emin and emax are None, assume limits are between twice the
        # hard-soft boundary and twice the contact boundary between two solar mass stars

        a_hs = grav * self.mbar / (sig0 ** 2.)  # pc
        a_max = 2. * a_hs
        a_min = (4.0 / 215.032) * 4.84814e-6  # pc

        if emin is None:
            e_min = grav * (self.mbar ** 2.) / (2.0 * a_max)  # Msun (km/s)**2
            e_min *= (1000.0 ** 2.)
            self.emin = e_min * msolar
        else:
            self.emin = emin

        if emax is None:
            e_max = grav * (self.mbar ** 2.) / (2.0 * a_min)  # Msun (km/s)**2
            e_max *= (1000.0 ** 2.)
            self.emax = e_max * msolar
        else:
            self.emax = emax

        if verbose:
            print('Sample Binary Energies between: ', self.emin, ' and ',
                  self.emax, ' J')

        self.q = q

        # Generate kick velocities for escaped stars and binaries
        vxkick = np.zeros(self.nstar)
        vykick = np.zeros(self.nstar)
        vzkick = np.zeros(self.nstar)

        vxkickb = np.zeros(self.nstar)
        vykickb = np.zeros(self.nstar)
        vzkickb = np.zeros(self.nstar)

        self.vesc = np.array([])

        self.dr = np.array([])

        nescape = 0

        self.mstar = np.zeros(self.nstar)
        self.mb1 = np.zeros(self.nstar)
        self.mb2 = np.zeros(self.nstar)
        self.eb = np.zeros(self.nstar)
        self.e0 = np.zeros(self.nstar)

        if binaries:
            self.binaries = True
            self.bindx = np.zeros(self.nstar, dtype=bool)
            self.vescb = np.array([])
        else:
            self.binaries = False

        if self.timing:
            dttime = time.time()

        while nescape < self.nstar:

            if m1 is None:
                if masses is None:
                    ms = self._power_law_distribution_function(1, self.alpha,
                                                               self.mmin,
                                                               self.mmax)
                else:
                    ms = np.random.choice(self.masses, 1)
            elif isinstance(m1, float) or isinstance(m1, int):
                ms = m1
            else:
                ms = np.random.choice(m1, 1)

            if m2a is None:
                if masses is None:
                    m_a = self._power_law_distribution_function(1, self.alpha,
                                                                self.mmin,
                                                                self.mmax)
                else:
                    m_a = np.random.choice(self.masses, 1)
            elif isinstance(m2a, float) or isinstance(m2a, int):
                m_a = m2a
            else:
                m_a = np.random.choice(m2a, 1)

            if m2b is None:
                if masses is None:
                    m_b = self._power_law_distribution_function(1, self.alpha,
                                                                self.mmin,
                                                                self.mmax)
                else:
                    m_b = np.random.choice(self.masses, 1)
            elif isinstance(m2b, float) or isinstance(m2b, int):
                m_b = m2b
            else:
                m_b = np.random.choice(m2b, 1)

            mb = m_a + m_b
            M = ms + mb

            prob = self._prob_three_body_escape(ms, m_a, m_b, self.q)  # will be new

            if np.random.rand() < prob:

                vxs, vys, vzs = np.random.normal(self.mu0, self.sig0, 3) # binary 1
                vstar = np.sqrt(vxs ** 2. + vys ** 2. + vzs ** 2.)
                vxb, vyb, vzb = np.random.normal(self.mu0, self.sig0, 3) # binary 2
                vbin = np.sqrt(vxb ** 2. + vyb ** 2. + vzb ** 2.)

                rdot = np.sqrt(
                    (vxs - vxb) ** 2. + (vys - vyb) ** 2. + (vzs - vzb) ** 2.)

                ebin, semi = self._sample_binding_energy(m_a, m_b, balpha,
                                                         self.emin, self.emax) # need new one for second binary

                if rsample:

                    rs = a_max / 2. + np.random.rand() * (
                                nrsep * self.rsep / 2. - a_max / 2.)
                    phis = 2.0 * np.pi * np.random.rand()
                    thetas = np.arccos(1.0 - 2.0 * np.random.rand())

                    xs = rs * np.cos(phis) * np.sin(thetas)
                    ys = rs * np.sin(phis) * np.sin(thetas)
                    zs = rs * np.cos(thetas)

                    rb = a_max / 2. + np.random.rand() * (
                                nrsep * self.rsep / 2. - a_max / 2.)
                    phib = 2.0 * np.pi * np.random.rand()
                    thetab = np.arccos(1.0 - 2.0 * np.random.rand())

                    xb = rb * np.cos(phib) * np.sin(thetab)
                    yb = rb * np.sin(phib) * np.sin(thetab)
                    zb = rb * np.cos(thetab)

                    dr = np.sqrt(
                        (xs - xb) ** 2. + (ys - yb) ** 2. + (zs - zb) ** 2.)

                    e0 = 0.5 * (mb * ms / M) * (
                                rdot ** 2.) - grav * ms * mb / dr + ebin # calculating energy
                else:
                    e0 = 0.5 * (mb * ms / M) * (
                                rdot ** 2.) - grav * ms * mb / self.rsep + ebin  # binding energy of binary
                    dr = self.rsep

                vs = self._sample_escape_velocity(e0, ms, mb, npeak, nrandom) # new function

                if vs > self.vesc0:  # refactor for first binary

                    self.vesc = np.append(self.vesc, vs)
                    self.dr = np.append(self.dr, dr)
                    vxkick[nescape] = vs * (vxs / vstar)
                    vykick[nescape] = vs * (vys / vstar)
                    vzkick[nescape] = vs * (vzs / vstar)

                    if binaries:  # refactor for second binary
                        # Check to see if recoil binary will also escape
                        # Binary kick velocity is calculated assuming total linear momentum of system sums to zero

                        pxi = ms * vxs + mb * vxb
                        pyi = ms * vys + mb * vyb
                        pzi = ms * vzs + mb * vzb

                        vxkickb[nescape] = (pxi - ms * vxkick[nescape]) / mb
                        vykickb[nescape] = (pyi - ms * vykick[nescape]) / mb
                        vzkickb[nescape] = (pzi - ms * vzkick[nescape]) / mb

                        vsb = np.sqrt(
                            vxkickb[nescape] ** 2. + vykickb[nescape] ** 2. +
                            vzkickb[nescape] ** 2.)

                        self.vescb = np.append(self.vescb, vsb)

                        if vsb > self.vesc0:
                            self.bindx[nescape] = True

                    self.mstar[nescape] = ms
                    self.mb1[nescape] = m_a
                    self.mb2[nescape] = m_b
                    self.eb[nescape] = ebin
                    self.e0[nescape] = e0

                    nescape += 1

                if verbose:
                    print('Sampling: ', nescape, prob, vs, self.vesc0)

        if self.timing:
            print(nescape, ' three body encounters simulated in ',
                  time.time() - dttime, ' s')

        if initialize:
            self.oi = self._initialize_orbits(vxkick, vykick, vzkick, False,
                                              verbose, **kwargs)
            self.of = None
            if binaries:
                self.obi = self._initialize_orbits(vxkickb, vykickb, vzkickb,
                                                   False, verbose, **kwargs)
                self.obf = None

            if binaries:
                return self.oi, self.obi
            else:
                return self.oi

        else:
            self.oi, self.of = self._integrate_orbits(vxkick, vykick, vzkick,
                                                      False, verbose, **kwargs)

            if binaries:
                self.obi, self.obf = self._integrate_orbits(vxkickb, vykickb,
                                                            vzkickb, binaries,
                                                            verbose, **kwargs)

            if binaries:
                return self.of, self.obf
            else:
                return self.of

    def _integrate_orbits(self, vxkick, vykick, vzkick, binaries=False,
                          verbose=False, **kwargs):

        # Set integration method (see https://docs.galpy.org/en/v1.7.2/orbit.html)
        method = kwargs.get('method', None)
        # Add a minimal offset to prevent stars from being initialized at r=0 in a the cluster.
        offset = kwargs.get('offset', 1e-9)
        xoffsets = np.random.normal(0.0, offset, len(vxkick))
        yoffsets = np.random.normal(0.0, offset, len(vykick))
        zoffsets = np.random.normal(0.0, offset, len(vzkick))

        Re0, phie0, ze0, vRe0, vTe0, vze0 = np.array([]), np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([])

        # Initial and final positions and velocities
        vxvv_i = []
        vxvv_f = []

        if self.timing:
            dttottime = time.time()

        for i in range(0, self.nstar):
            if self.timing:
                dttime = time.time()

            xi, yi, zi = self.o.x(self.tesc[i] / self.to) + xoffsets[
                i], self.o.y(self.tesc[i] / self.to) + yoffsets[i], self.o.z(
                self.tesc[i] / self.to) + zoffsets[i]
            vxi = vxkick[i] + self.o.vx(self.tesc[i] / self.to)
            vyi = vykick[i] + self.o.vy(self.tesc[i] / self.to)
            vzi = vzkick[i] + self.o.vz(self.tesc[i] / self.to)

            if verbose:
                print(i, self.tesc[i], xi, yi, zi, vxi, vyi, vzi)

            # Save initial positions and velocities

            Ri, phii, zi = coords.rect_to_cyl(xi, yi, zi)
            vRi, vTi, vzi = coords.rect_to_cyl_vec(vxi, vyi, vzi, xi, yi, zi)

            vxvv_i.append(
                [Ri / self.ro, vRi / self.vo, vTi / self.vo, zi / self.ro,
                 vzi / self.vo, phii])

            if not binaries or (binaries and self.bindx[i]):
                # Integrate orbit from tesc to 0.
                os = Orbit(vxvv_i[-1], ro=self.ro, vo=self.vo,
                           solarmotion=[-11.1, 24.0, 7.25])

                ntstep = kwargs.get('ntstep', 10000)
                ts = np.linspace(self.tesc[i] / self.to, 0., ntstep)

                if method is None:
                    os.integrate(ts, self.pot)
                else:
                    os.integrate(ts, self.pot, method=method)

                # Save final positions and velocities
                vxvv_f.append([os.R(0.) / self.ro, os.vR(0.) / self.vo,
                               os.vT(0.) / self.vo, os.z(0.) / self.ro,
                               os.vz(0.) / self.vo, os.phi(0.)])

            elif binaries and not self.bindx[i]:
                vxvv_f.append([self.o.R(0.) / self.ro, self.o.vR(0.) / self.vo,
                               self.o.vT(0.) / self.vo, self.o.z(0.) / self.ro,
                               self.o.vz(0.) / self.vo, self.o.phi(0.)])

            if self.timing:
                print('ORBIT ', i, ' INTEGRATED FROM %f WITH VK= %f KM/S IN' % (
                self.tesc[i], self.vesc[i]), time.time() - dttime,
                      ' s (Rp=%f)' % os.rperi())

        if self.timing:
            print('ALL ORBITS INTEGRATED IN', time.time() - dttottime, ' s')

        # Save initial and final positions and velocities of kicked stars at t=0 in orbit objects
        oi = Orbit(vxvv_i, ro=self.ro, vo=self.vo,
                   solarmotion=[-11.1, 24.0, 7.25])
        of = Orbit(vxvv_f, ro=self.ro, vo=self.vo,
                   solarmotion=[-11.1, 24.0, 7.25])

        return oi, of

    def _initialize_orbits(self, vxkick, vykick, vzkick, binaries=False,
                           verbose=False, **kwargs):

        # Set integration method (see https://docs.galpy.org/en/v1.7.2/orbit.html)
        method = kwargs.get('method', None)
        # Add a minimal offset to prevent stars from being initialized at r=0 in a the cluster.
        offset = kwargs.get('offset', 1e-9)
        xoffsets = np.random.normal(0.0, offset, len(vxkick))
        yoffsets = np.random.normal(0.0, offset, len(vykick))
        zoffsets = np.random.normal(0.0, offset, len(vzkick))

        Re0, phie0, ze0, vRe0, vTe0, vze0 = np.array([]), np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([])

        # Initial and final positions and velocities
        vxvv_i = []
        vxvv_f = []

        if self.timing:
            dttottime = time.time()

        for i in range(0, self.nstar):
            if self.timing:
                dttime = time.time()

            xi, yi, zi = self.o.x(self.tesc[i] / self.to) + xoffsets[
                i], self.o.y(self.tesc[i] / self.to) + yoffsets[i], self.o.z(
                self.tesc[i] / self.to) + zoffsets[i]
            vxi = vxkick[i] + self.o.vx(self.tesc[i] / self.to)
            vyi = vykick[i] + self.o.vy(self.tesc[i] / self.to)
            vzi = vzkick[i] + self.o.vz(self.tesc[i] / self.to)

            if verbose:
                print(i, self.tesc[i], xi, yi, zi, vxi, vyi, vzi)

            # Save initial positions and velocities

            Ri, phii, zi = coords.rect_to_cyl(xi, yi, zi)
            vRi, vTi, vzi = coords.rect_to_cyl_vec(vxi, vyi, vzi, xi, yi, zi)

            vxvv_i.append(
                [Ri / self.ro, vRi / self.vo, vTi / self.vo, zi / self.ro,
                 vzi / self.vo, phii])

        # Save initial and final positions and velocities of kicked stars at t=0 in orbit objects
        oi = Orbit(vxvv_i, ro=self.ro, vo=self.vo,
                   solarmotion=[-11.1, 24.0, 7.25])

        return oi

    def _prob_three_body_escape(self, ms, m_a, m_b, q): # find new one for 4 body (no need)

        # Equation 7.23
        prob = (ms ** q) / (ms ** q + m_a ** q + m_b ** q)
        return prob

    def _sample_binding_energy(self, mb1, mb2, balpha, emin, emax):
        # Opik's Law
        # Default binding energy distribution is:
        # power law of slope -1
        # Between 10.0**36 and 10.0**40 J

        grav = 4.302e-3  # pc/Msun (km/s)^2

        if isinstance(mb1, float) or isinstance(mb1, int):
            n = 1
        else:
            n = len(mb1)

        ebin_si = self._power_law_distribution_function(n, balpha, emin,
                                                        emax)  # Joules = kg (m/s)^2
        ebin = ebin_si / 1.9891e30  # Msun (m/s)^2
        ebin /= (1000.0 * 1000.0)  # Msun (km/s)^2

        # Log normal a:
        semi_pc = (0.5 * grav * mb1 * mb2) / ebin
        semi_au = semi_pc / 4.84814e-6

        semi = semi_pc

        return ebin, semi

    def _sample_escape_velocity(self, e0, ms, mb, npeak=5, nrandom=1000): # new version for 4 body
        # randomly sample between npeak*vs_peak

        vs_peak = self._escape_velocity_distribution_peak(e0, ms, mb)
        match = False

        while not match:
            vstemp = np.random.rand(nrandom) * npeak * vs_peak
            amptemp = np.random.rand(nrandom) * vs_peak

            aindx = amptemp < self._escape_velocity_distribution(
                np.ones(nrandom) * vstemp, np.ones(nrandom) * e0,
                np.ones(nrandom) * ms, np.ones(nrandom) * mb)

            if np.sum(aindx) > 0:
                vs = vstemp[aindx][0]
                match = True

        return vs

    def _escape_velocity_distribution(self, vs, e0, ms, mb):
        # Equation 7.19
        M = ms + mb
        fv = (3.5 * (np.fabs(e0) ** (7. / 2.)) * ms * M / mb) * vs / (
                    (np.fabs(e0) + 0.5 * (ms * M / mb) * (vs ** 2.)) ** (
                        9. / 2.))
        return fv

    def _escape_velocity_distribution_peak(self, e0, ms, mb):
        M = ms + mb
        vs_peak = 0.5 * np.sqrt((M - ms) / (ms * M)) * np.sqrt(np.fabs(e0))

        return vs_peak

# new functions --------------------------------------------------------------
    def _escape_velocity_distribution_four_body(self, n, vs, e0, ms, mb):
        # Equation 24 from Leigh et al., 2021
        M = ms + mb
        fv = (n * (np.fabs(e0) ** (n - 1)) * ms * M / mb) * vs / \
             ((np.fabs(e0) + 0.5 * (ms * M / mb) * (vs ** 2.)) ** n)
        return fv

    def _escape_velocity_distribution_peak_four_body(self, n, e0, ms, mb):
        # Equation 25 from Leigh et al., 2021
        M = ms + mb
        epsilon = (n - 0.5) ** (-0.5)
        vs_peak = epsilon * ((M - ms) / (ms * M)) ** (0.5) * (np.fabs(e0)) ** (
            0.5)

        return vs_peak

# -----------------------------------------------------------------------------

    def sample_uniform(self, tdisrupt=1000., rate=1., nstar=None, vmin=0.,
                       vmax=500., verbose=False, **kwargs):
        """ A function for sampling a uniform core ejection distribution function

        Parameters
        ----------

        tdisrupt : float
            time over which sampling begins (Myr)
        rate : float
            ejection rate (default 1 per Myr)
        nstar : float
            if set, nstar stars will be ejected randomly from tdisrupt to 0 Myr. Rate is recalculated. (default : None)
        vmin : float
            minimum kick velocity
        vmax : float
            maximum kick velocity
        verbose : bool
            print additional information to screen (default: False)

        Returns
        ----------
        of : orbit
            galpy orbit instance for kicked stars


        History
        -------
        2021 - Written - Grandin/Webb (UofT)

        """

        self.tdisrupt = tdisrupt

        # Select escape times
        # If nstar is not None, randomly select escapers between tstart and tend
        if nstar is not None:
            self.nstar = nstar
            self.rate = nstar / self.tdisrupt
        else:
            self.rate = rate
            self.nstar = self.tdisrupt * rate

        self.tesc = -1. * self.tdisrupt * np.random.rand(self.nstar)

        ntstep = kwargs.get('ntstep', 10000)
        ts = np.linspace(0., -1. * self.tdisrupt / self.to, ntstep)
        self.o.integrate(ts, self.mwpot)

        if self.gcpot is None:
            self.pot = self.mwpot
        else:
            moving_pot = MovingObjectPotential(self.o, self.gcpot, ro=self.ro,
                                               vo=self.vo)
            self.pot = [self.mwpot, moving_pot]

        # Generate kick velocities for escaped stars and binaries
        self.vesc = vmin + (vmax - vmin) * np.random.rand(self.nstar)

        # Assume a random direction
        vxs = np.random.normal(0, 1., self.nstar)
        vys = np.random.normal(0, 1., self.nstar)
        vzs = np.random.normal(0, 1., self.nstar)

        vstar = np.sqrt(vxs ** 2. + vys ** 2. + vzs ** 2.)

        vxkick = self.vesc * (vxs / vstar)
        vykick = self.vesc * (vys / vstar)
        vzkick = self.vesc * (vzs / vstar)

        self.oi, self.of = self._integrate_orbits(vxkick, vykick, vzkick,
                                                  **kwargs)

        return self.of

    def sample_gaussian(self, tdisrupt=1000., rate=1., nstar=None, vmean=100.,
                        vsig=10., verbose=False, **kwargs):
        """ A function for sampling a uniform core ejection distribution function

        Parameters
        ----------

        tdisrupt : float
            time over which sampling begins (Myr)
        rate : float
            ejection rate (default 1 per Myr)
        nstar : float
            if set, nstar stars will be ejected randomly from tdisrupt to 0 Myr. Rate is recalculated. (default : None)
        vmean : float
            average kick velocity
        vsig : float
            standard deviation of kick velocity distribution
        verbose : bool
            print additional information to screen (default: False)

        Returns
        ----------
        of : orbit
            galpy orbit instance for kicked stars


        History
        -------
        2022 - Written - Grondin/Webb (UofT)

        """

        self.tdisrupt = tdisrupt

        # Select escape times
        # If nstar is not None, randomly select escapers between tstart and tend
        if nstar is not None:
            self.nstar = nstar
            self.rate = nstar / self.tdisrupt
        else:
            self.rate = rate
            self.nstar = self.tdisrupt * rate

        self.tesc = -1. * self.tdisrupt * np.random.rand(self.nstar)

        ntstep = kwargs.get('ntstep', 10000)
        ts = np.linspace(0., -1. * self.tdisrupt / self.to, ntstep)
        self.o.integrate(ts, self.mwpot)

        if self.gcpot is None:
            self.pot = self.mwpot
        else:
            moving_pot = MovingObjectPotential(self.o, self.gcpot, ro=self.ro,
                                               vo=self.vo)
            self.pot = [self.mwpot, moving_pot]

        # Generate kick velocities for escaped stars and binaries
        self.vesc = np.random.normal(vmean, vsig, nstar)

        # Assume a random direction
        vxs = np.random.normal(vmean, vsig, self.nstar)
        vys = np.random.normal(vmean, vsig, self.nstar)
        vzs = np.random.normal(vmean, vsig, self.nstar)

        vstar = np.sqrt(vxs ** 2. + vys ** 2. + vzs ** 2.)

        vxkick = self.vesc * (vxs / vstar)
        vykick = self.vesc * (vys / vstar)
        vzkick = self.vesc * (vzs / vstar)

        self.oi, self.of = self._integrate_orbits(vxkick, vykick, vzkick,
                                                  **kwargs)

        return self.of

# might have to modify the below ----------------------------------------------

    def _power_law_distribution_function(self, n, alpha, xmin, xmax):

        eta = alpha + 1.

        if xmin == xmax:
            x = xmin
        elif alpha == 0:
            x = xmin + np.random.random(n) * (xmax - xmin)
        elif alpha > 0:
            x = xmin + np.random.power(eta, n) * (xmax - xmin)
        elif alpha < 0 and alpha != -1.:
            x = (xmin ** eta + (xmax ** eta - xmin ** eta) * np.random.rand(
                n)) ** (1. / eta)
        elif alpha == -1:
            x = np.log10(xmin) + np.random.random(n) * (
                        np.log10(xmax) - np.log10(xmin))
            x = 10.0 ** x

        if n == 1:
            return x
        else:
            return np.array(x)

    def _init_fig(self, xlim=(-20, 20), ylim=(-20, 20)):
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=xlim, ylim=ylim)
        self.ax.set_xlabel('X (kpc)')
        self.ax.set_ylabel('Y (kpc)')
        self.txt_title = self.ax.set_title('')
        self.line, = self.ax.plot([], [], lw=2)
        self.pt, = self.ax.plot([], [], '.')
        self.pt2, = self.ax.plot([], [], '.')

    def _set_data(self, gcdata, sdata, bdata):
        self.gcdata = gcdata
        self.sdata = sdata
        self.bdata = bdata

    def _ani_init(self):
        self.line.set_data([], [])
        self.pt.set_data([], [])
        self.pt2.set_data([], [])

        return self.line, self.pt, self.pt2

    def _ani_update(self, i):

        if i < 5:
            x = self.gcdata[0:i + 1, 0]
            y = self.gcdata[0:i + 1, 1]
        else:
            x = self.gcdata[i - 5:i + 1, 0]
            y = self.gcdata[i - 5:i + 1, 1]
        self.line.set_data(x, y)

        escindx = self.tesc / self.to <= self.ts[i]

        if np.sum(escindx) > 0:
            self.pt.set_data(self.sdata[i][0][escindx],
                             self.sdata[i][1][escindx])
        else:
            self.pt.set_data([], [])

        if self.binaries:
            if np.sum(escindx) > 0:
                self.pt2.set_data(self.bdata[i][0][escindx * self.bindx],
                                  self.bdata[i][1][escindx * self.bindx])
            else:
                self.pt2.set_data([], [])

        self.txt_title.set_text('%s' % str(self.ts[i] * self.to))

        return self.line, self.pt, self.pt2

# no need to change the below -------------------------------------------------

    def animate(self, frames=100, interval=50, xlim=(-20, 20), ylim=(-20, 20)):

        """Animate the ejection of stars from the cluster's core

        Parameters
        ----------

        frames : int
            number of frames to use for animation (default:100)
        interval : float
            time intercal between frames (default: 50 Myr)
        xlim : tuple
            xlimits for figure
        ylim : tuple
            ylimts for figure

        History
        -------
        2021 - Written - Webb (UofT)

        """

        self._init_fig(xlim, ylim)

        self.ts = np.linspace(-1. * self.tdisrupt / self.to, 0., frames)
        tsint = np.linspace(0., -1. * self.tdisrupt / self.to, 1000)
        self.of.integrate(tsint, self.pot)
        if self.binaries:
            self.obf.integrate(tsint, self.pot)

        gcdata = np.zeros(shape=(frames, 2))

        for i in range(0, frames):
            gcdata[i] = [self.o.x(self.ts[i]), self.o.y(self.ts[i])]

        sdata = np.zeros(shape=(frames, 2, self.nstar))

        for i in range(0, frames):
            sdata[i] = [self.of.x(self.ts[i]), self.of.y(self.ts[i])]

        if self.binaries:
            bdata = np.zeros(shape=(frames, 2, self.nstar))
            for i in range(0, frames):
                bdata[i] = [self.obf.x(self.ts[i]), self.obf.y(self.ts[i])]
        else:
            bdata = None

        self._set_data(gcdata, sdata, bdata)

        self.anim = animation.FuncAnimation(self.fig, self._ani_update,
                                            init_func=self._ani_init,
                                            frames=frames, interval=interval,
                                            blit=False)

# no need to change the below -------------------------------------------------

    def snapout(self, filename='corespray.dat', filenameb='coresprayb.dat'):
        """Output present day positions, velocities, escape times, and escape
           velocities of stars

        Parameters
        ----------

        filename: str
            file name to write data to (default: corespray.dat)
        filenameb: str
            file name to write binary data to (default: corespray.dat)
        History
        -------
        2021 - Written - Webb (UofT)

        """
        R = np.append(self.o.R(0.), self.of.R(0.))
        vR = np.append(self.o.vR(0.), self.of.vR(0.))
        vT = np.append(self.o.vT(0.), self.of.vT(0.))
        z = np.append(self.o.z(0.), self.of.z(0.))
        vz = np.append(self.o.vz(0.), self.of.vz(0.))
        phi = np.append(self.o.phi(0.), self.of.phi(0.))

        vesc = np.append(0., self.vesc)
        tesc = np.append(0., self.tesc)

        np.savetxt(filename,
                   np.column_stack([R, vR, vT, z, vz, phi, vesc, tesc]))

        if self.binaries:
            R = self.obf.R(0.)
            vR = self.obf.vR(0.)
            vT = self.obf.vT(0.)
            z = self.obf.z(0.)
            vz = self.obf.vz(0.)
            phi = self.obf.phi(0.)

            vesc = self.vescb
            tesc = self.tesc

            np.savetxt(filenameb, np.column_stack(
                [R, vR, vT, z, vz, phi, vesc, tesc, self.bindx]))
