""" The corespraydf_4body class

"""

__author__ = "Steffani Grondin & Jeremy J Webb & Ritik Kothari"

__all__ = [
    "corespraydf_4body",
]

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, PlummerPotential, KingPotential, \
    MovingObjectPotential
from galpy.util import conversion, coords
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

import time


class corespraydf_4body(object):
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
    2023 - Modified - Kothari (UofT)
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

        self.triples = False

        self.mt = None

        self.timing = timing

    # -----------------------------------------------------------------------------
    def sample_four_body(self, tdisrupt=1000., rate=1., nstar=None, mu0=0.,
                          sig0=10.0, vesc0=10.0, rho0=1., mmin=0.1, mmax=1.4,
                          alpha=-1.35, masses=None, m1a=None, m1b=None, m2a=None,
                          m2b=None, emin=None, emax=None, balpha=-1, q=-3, npeak=5.,
                          triples=False, verbose=False, **kwargs):
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
            average 1D velocity dispersions in the core (default 10.0 km/s)
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
        triples : bool
            keep track of triples that receive recoil kicks greater than the
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

        if triples:
            otf : orbit
                galpy orbit instance for recoil triples stars

        History
        -------
        2021 - Written - Grondin/Webb (UofT)
        2023 - Modified - Kothari (UofT)
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
            # Mean separation of star's in the core equal to twice the radius
            # of a sphere that contains one star
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

        # Generate kick velocities for escaped stars and triples
        vxkick = np.zeros(self.nstar)
        vykick = np.zeros(self.nstar)
        vzkick = np.zeros(self.nstar)

        vxkickt = np.zeros(self.nstar)
        vykickt = np.zeros(self.nstar)
        vzkickt = np.zeros(self.nstar)

        self.vesc = np.array([])

        self.dr = np.array([])

        nescape = 0

        self.mstar = np.zeros(self.nstar)
        self.mb1 = np.zeros(self.nstar)
        self.mb2 = np.zeros(self.nstar)
        self.mt = np.zeros(self.nstar)
        self.eb1 = np.zeros(self.nstar)  # energy of first binary
        self.eb2 = np.zeros(self.nstar)  # energy of second binary
        self.e0 = np.zeros(self.nstar)   # initial energy of system

        if triples:
            self.triples = True
            self.tindx = np.zeros(self.nstar, dtype=bool)
            self.vesct = np.array([])
        else:
            self.triples = False

        if self.timing:
            dttime = time.time()

        while nescape < self.nstar:

            if m1a is None:  # determining escaper mass
                if masses is None:
                    ms = self._power_law_distribution_function(1, self.alpha,
                                                               self.mmin,
                                                               self.mmax)
                else:
                    ms = np.random.choice(self.masses, 1)
            elif isinstance(m1a, float) or isinstance(m1a, int):
                ms = m1a
            else:
                ms = np.random.choice(m1a, 1)

            if m2a is None:  # determining mass of escaper's partner
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

            if m1b is None:  # determining mass of first star in second binary
                if masses is None:
                    m_b = self._power_law_distribution_function(1, self.alpha,
                                                                self.mmin,
                                                                self.mmax)
                else:
                    m_b = np.random.choice(self.masses, 1)
            elif isinstance(m1b, float) or isinstance(m1b, int):
                m_b = m1b
            else:
                m_b = np.random.choice(m1b, 1)

            if m2b is None:  # determining mass of second star in second binary
                if masses is None:
                    m_c = self._power_law_distribution_function(1, self.alpha,
                                                                self.mmin,
                                                                self.mmax)
                else:
                    m_c = np.random.choice(self.masses, 1)
            elif isinstance(m2b, float) or isinstance(m2b, int):
                m_c = m2b
            else:
                m_c = np.random.choice(m2b, 1)

            # if escaper mass is not given, use the below logic to assign lowest
            # mass star as escaper
            if m1a is None:
                # creates a list of the four masses sampled from the power law
                # distribution
                mass_lst = [ms, m_a, m_b, m_c]

                # takes the lowest mass of the four to be the escaper star
                ms = min(mass_lst)

                # removes the escaper from the list while leaving the other
                # masses in the list (without ordering the masses)
                mass_lst.remove(min(mass_lst))

                # assigns the other masses to the remaining values in the order
                # from the original list
                m_a, m_b, m_c = mass_lst

            mt = m_a + m_b + m_c
            # M = ms + mt

            prob = self._prob_three_body_escape(ms, m_b, m_c, self.q)
            # because mass on wide orbit around remaining binary is negligible

            if np.random.rand() < prob:

                vxs, vys, vzs = np.random.normal(self.mu0, self.sig0, 3) # single star
                vstar = np.sqrt(vxs ** 2. + vys ** 2. + vzs ** 2.)
                vxt, vyt, vzt = np.random.normal(self.mu0, self.sig0, 3) # triple
                vtrip = np.sqrt(vxt ** 2. + vyt ** 2. + vzt ** 2.)

                rdot = np.sqrt(
                    (vxs - vxt) ** 2. + (vys - vyt) ** 2. + (vzs - vzt) ** 2.)

                # binding energy for first binary containing escaper star
                ebin1, semi1 = self._sample_binding_energy(ms, m_a, balpha,
                                                         self.emin, self.emax)

                # binding energy for second binary
                ebin2, semi2 = self._sample_binding_energy(m_b, m_c, balpha,
                                                          self.emin, self.emax)

                if rsample:

                    rs = a_max / 2. + np.random.rand() * (
                            nrsep * self.rsep / 2. - a_max / 2.)
                    phis = 2.0 * np.pi * np.random.rand()
                    thetas = np.arccos(1.0 - 2.0 * np.random.rand())

                    xs = rs * np.cos(phis) * np.sin(thetas)
                    ys = rs * np.sin(phis) * np.sin(thetas)
                    zs = rs * np.cos(thetas)

                    rt = a_max / 2. + np.random.rand() * (
                            nrsep * self.rsep / 2. - a_max / 2.)
                    phit = 2.0 * np.pi * np.random.rand()
                    thetat = np.arccos(1.0 - 2.0 * np.random.rand())

                    xt = rt * np.cos(phit) * np.sin(thetat)
                    yt = rt * np.sin(phit) * np.sin(thetat)
                    zt = rt * np.cos(thetat)

                    dr = np.sqrt(
                        (xs - xt) ** 2. + (ys - yt) ** 2. + (zs - zt) ** 2.)

                else:
                    dr = self.rsep

                initial_energy_1 = (1/2)*(ms + m_a)*rdot**2

                potential_energy_1 = - grav * (m_b + m_c) / dr

                e0 = initial_energy_1 + potential_energy_1 + ebin1 + ebin2

                vs = self._sample_escape_velocity_four_body(e0, ms, (m_b + m_c),
                                                            int(npeak), nrandom)

                if vs > self.vesc0:  # refactor for first binary (should be kept the same for escaper)

                    self.vesc = np.append(self.vesc, vs)
                    self.dr = np.append(self.dr, dr)
                    vxkick[nescape] = vs * (vxs / vstar)
                    vykick[nescape] = vs * (vys / vstar)
                    vzkick[nescape] = vs * (vzs / vstar)

                    if triples:  # refactor for second binary (considering triples now)
                        # Check to see if recoil triple will also escape
                        # Triple kick velocity is calculated assuming total linear momentum of system sums to zero

                        pxi = ms * vxs + mt * vxt
                        pyi = ms * vys + mt * vyt
                        pzi = ms * vzs + mt * vzt

                        vxkickt[nescape] = (pxi - ms * vxkick[nescape]) / mt
                        vykickt[nescape] = (pyi - ms * vykick[nescape]) / mt
                        vzkickt[nescape] = (pzi - ms * vzkick[nescape]) / mt

                        vst = np.sqrt(
                            vxkickt[nescape] ** 2. + vykickt[nescape] ** 2. +
                            vzkickt[nescape] ** 2.)

                        self.vesct = np.append(self.vesct, vst)

                        if vst > self.vesc0:
                            self.tindx[nescape] = True

                    self.mstar[nescape] = ms
                    self.mb1[nescape] = m_b
                    self.mb2[nescape] = m_c
                    self.mt[nescape] = m_a
                    # self.eb[nescape] = ebin2
                    # self.ebt[nescape] = ebin1
                    # self.eb2[nescape] = ebin2
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
            if triples:
                self.oti = self._initialize_orbits(vxkickt, vykickt, vzkickt,
                                                   False, verbose, **kwargs)
                self.otf = None

            if triples:
                return self.oi, self.oti
            else:
                return self.oi

        else:
            self.oi, self.of = self._integrate_orbits(vxkick, vykick, vzkick,
                                                      False, verbose, **kwargs)

            if triples:
                self.oti, self.otf = self._integrate_orbits(vxkickt, vykickt,
                                                            vzkickt, triples,
                                                            verbose, **kwargs)

            if triples:
                return self.of, self.otf
            else:
                return self.of

    def _integrate_orbits(self, vxkick, vykick, vzkick, triples=False,
                          verbose=False, **kwargs):

        # Set integration method (see https://docs.galpy.org/en/v1.7.2/orbit.html)
        global os
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

            if not triples or (triples and self.tindx[i]):
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

            elif triples and not self.tindx[i]:
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

    def _initialize_orbits(self, vxkick, vykick, vzkick, triples=False,
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

    # return type float
    def _sample_escape_velocity_four_body(self, e0, mb1, mb2, npeak=5, nrandom=1000):
        # randomly sample between npeak * vs_peak

        vs_peak = self._escape_velocity_distribution_peak_four_body(e0, mb1, mb2)
        match = False

        while not match:
            vstemp = np.random.rand(nrandom) * npeak * vs_peak
            amptemp = np.random.rand(nrandom) * vs_peak

            aindx = amptemp < self._escape_velocity_distribution_four_body(
                np.ones(nrandom) * vstemp, np.ones(nrandom) * e0,
                np.ones(nrandom) * mb1, np.ones(nrandom) * mb2)

            if np.sum(aindx) > 0:
                vs = vstemp[aindx][0]
                match = True

        return vs

    def _escape_velocity_distribution_four_body(self, vs, e0, mb1, mb2, n=3):
        # Equation 24 from Leigh et al., 2021
        M = mb1 + mb2
        fv = ((n - 1) * (np.fabs(e0) ** (n - 1)) * mb1 * M / mb2) * vs / \
             ((np.fabs(e0) + 0.5 * (mb1 * M / mb2) * (vs ** 2.)) ** n)
        return fv

    def _escape_velocity_distribution_peak_four_body(self, e0, mb1, mb2, n=3):
        # Equation 25 from Leigh et al., 2021
        M = mb1 + mb2
        epsilon = (n - 0.5) ** (-0.5)
        vs_peak = epsilon * ((M - mb1) / (mb2 * M)) ** 0.5 * (np.fabs(e0)) ** 0.5

        return vs_peak

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
        2021 - Written - Grondin/Webb (UofT)

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

        if self.triples:
            if np.sum(escindx) > 0:
                self.pt2.set_data(self.bdata[i][0][escindx * self.tindx],
                                  self.bdata[i][1][escindx * self.tindx])
            else:
                self.pt2.set_data([], [])

        self.txt_title.set_text('%s' % str(self.ts[i] * self.to))

        return self.line, self.pt, self.pt2


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
        if self.triples:
            self.otf.integrate(tsint, self.pot)

        gcdata = np.zeros(shape=(frames, 2))

        for i in range(0, frames):
            gcdata[i] = [self.o.x(self.ts[i]), self.o.y(self.ts[i])]

        sdata = np.zeros(shape=(frames, 2, self.nstar))

        for i in range(0, frames):
            sdata[i] = [self.of.x(self.ts[i]), self.of.y(self.ts[i])]

        if self.triples:
            bdata = np.zeros(shape=(frames, 2, self.nstar))
            for i in range(0, frames):
                bdata[i] = [self.otf.x(self.ts[i]), self.otf.y(self.ts[i])]
        else:
            bdata = None

        self._set_data(gcdata, sdata, bdata)

        self.anim = animation.FuncAnimation(self.fig, self._ani_update,
                                            init_func=self._ani_init,
                                            frames=frames, interval=interval,
                                            blit=False)


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

        if self.triples:
            R = self.otf.R(0.)
            vR = self.otf.vR(0.)
            vT = self.otf.vT(0.)
            z = self.otf.z(0.)
            vz = self.otf.vz(0.)
            phi = self.otf.phi(0.)

            vesc = self.vesct
            tesc = self.tesc

            np.savetxt(filenameb, np.column_stack(
                [R, vR, vT, z, vz, phi, vesc, tesc, self.tindx]))


