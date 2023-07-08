""" The corespraydf class (for 4 bodies)

"""

__author__ = "Steffani Grondin, Jeremy J Webb, Ritik Kothari"

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

""" The following file contains comments made by Ritik as notes taken that 
    analyze the corespray code.
"""


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
	verbose : bool
			print additional information to screen (default: False)

	History
	-------
	2021 - Written - Grondin (UofT)
	2023 - Modified - Kothari (UofT)

	"""

    # initializing corespray parameters
    def __init__(self, gcorbit, pot=MWPotential2014, mgc=None, rgc=None,
                 W0=None, ro=8., vo=220., verbose=False):

        # determines whether given GC is named or not
        if isinstance(gcorbit, str):
            self.gcname = gcorbit
            # not quite sure what the parameter 'o' is in this case
            self.o = Orbit.from_name(self.gcname, ro=ro, vo=vo,
                                     solarmotion=[-11.1, 24.0, 7.25])
        else:
            self.gcname = 'unknown'
            self.o = gcorbit

        self.ro, self.vo = ro, vo
        # initial age and mass of GC
        self.to = conversion.time_in_Gyr(ro=self.ro, vo=self.vo) * 1000.
        self.mo = conversion.mass_in_msol(ro=self.ro, vo=self.vo)

        self.mwpot = pot

        # if the GC has no mass then it cannot have a potential either
        if mgc is None:
            self.gcpot = None
        else:
            if W0 is None:
                # why is ra the result of dividing the half mass radius of the
                # GC by 1.3?
                ra = rgc / 1.3
                # As mentioned earlier, W0 is PlummerPotential if W0 = None
                # PlummerPotential and KingPotential are both functions from the
                # galpy package
                self.gcpot = PlummerPotential(mgc / self.mo, ra / self.ro,
                                              ro=self.ro, vo=self.vo)
            else:
                self.gcpot = KingPotential(W0, mgc / self.mo, rgc / self.ro,
                                           ro=self.ro, vo=self.vo)

        self.triplets = False

    # -----------------------------------------------------------------------------

    def sample_four_body(self, tdisrupt=1000., rate=1., nstar=None, mu0=0.,
                         sig0=10.0, vesc0=10.0, rho0=1., mmin=0.1, mmax=1.4,
                         alpha=-1.35, masses=None, m1a=None, m1b=None,
                         m2a=None, m2b=None, emin=None, emax=None, q=-3, npeak=5.,
                         triplets=False, verbose=False):
        """ A function for sampling the four-body interaction core ejection
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
		masses: int
		    number of masses in the interaction (default: None)
		m1a: float
		    mass of first star in first binary (default: None)
		m1b: float
		    mass of second star in first binary (default: None)
		m2a: float
		    mass of first star in second binary (default: None)
		m2b: float
		    mass of second star in second binary (default: None)

-------- the below parameters are not used in this function (maybe) ------------

		mgc : float
			globular cluster mass - needed if cluster's potential is to be
			included in orbit integration of escapers (default: None)
		rgc : float
			half-mass radius of globular cluster (assuming Plummer potential)
			or tidal radius of globular cluster (assuming King potential)
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
		emin : float
			minimum binary energy (default: None)
		emax : float
			maximum binary energy (default: None)
		q : float
			exponent for calculating probability of stellar escape from
			three-body system (#Equation 7.23) (default: -3)
		npeak : float
			when sampling kick velocity distribution function, sampling range
			will be from 0 to npeak*vpeak, where vpeak is the peak in the
			distribution function (default: 5)
		triplets : bool
			keep track of triplets that receive recoil kicks greater than the
			cluster's escape velocity (default : False)
		verbose : bool
			print additional information to screen (default: False)

		Returns
		----------
		of : orbit
			galpy orbit instance for kicked stars

		if triplets:
			obf : orbit
				galpy orbit instance for recoil triplet stars

		History
		-------
		2021 - Written - Grondin/Webb (UofT)
		2023 - Modified - Kothari (UofT)
		"""

        grav = 4.302e-3  # pc/Msun (km/s)^2
        msolar = 1.9891e30

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

        ts = np.linspace(0., -1. * self.tdisrupt / self.to, 1000)
        self.o.integrate(ts, self.mwpot)

        if self.gcpot is None:
            self.pot = self.mwpot
        else:
            # galpy function that returns None
            moving_pot = MovingObjectPotential(self.o, self.gcpot, ro=self.ro,
                                               vo=self.vo)
            self.pot = [self.mwpot, moving_pot]

        # -----------------------------------------------------------------------------

        self.mu0, self.sig0, self.vesc0, self.rho0 = mu0, sig0, vesc0, rho0

        self.mmin, self.mmax, self.alpha = mmin, mmax, alpha

        # Mean separation of stars in the core equal to twice the radius of a
        # sphere that contains one star
        # Assume all stars in the core have mass equal to the mean mass
        masses = self._power_law_distribution_function(1000, self.alpha,
                                                       self.mmin, self.mmax)
        self.mbar = np.mean(masses)
        self.rsep = ((self.mbar / self.rho0) / (4. * np.pi / 3.)) ** (1. / 3.)

        # Limits of binary energy distribution
        # If emin and emax (minimum/maximum binary energy) are None,
        # assume limits are between twice the hard-soft boundary and
        # twice the contact boundary between two solar mass stars

        if emin is None:
            a_hs = grav * self.mbar / (sig0 ** 2.)  # pc
            a_max = 2. * a_hs
            e_min = grav * (self.mbar ** 2.) / (2.0 * a_max)  # Msun (km/s)**2
            e_min *= (1000.0 ** 2.)

            self.emin = e_min * msolar

        else:
            self.emin = emin

        if emax is None:
            a_min = (4.0 / 215.032) * 4.84814e-6  # pc
            e_max = grav * (self.mbar ** 2.) / (2.0 * a_min)  # Msun (km/s)**2
            e_max *= (1000.0 ** 2.)

            self.emax = e_max * msolar

        else:
            self.emax = emax

        if verbose:
            print('Sample Binary Energies between: ', self.emin, ' and ',
                  self.emax, ' J')

        self.q = q

        # Generate kick velocities for escaped stars and triplets
        vxkick = np.zeros(self.nstar)
        vykick = np.zeros(self.nstar)
        vzkick = np.zeros(self.nstar)

        vxkickt = np.zeros(self.nstar)  # triplets
        vykickt = np.zeros(self.nstar)
        vzkickt = np.zeros(self.nstar)

        self.vesc = np.array([])

        nescape = 0

        self.mstar = np.zeros(self.nstar)
        self.mb1 = np.zeros(self.nstar)
        self.mb2 = np.zeros(self.nstar)
        self.eb = np.zeros(self.nstar)

        if triplets:
            self.triplets = True
            self.bindx = np.zeros(self.nstar, dtype=bool)
            self.vescb = np.array([])
        else:
            self.triplets = False

        while nescape < self.nstar:
            mass_array = self._power_law_distribution_function(4, self.alpha,
                                                                 self.mmin,
                                                                 self.mmax)
            ms = min(mass_array)
            mass_array.remove(min(mass_array))
            m_a, m_b, m_c = mass_array
            mb1 = ms + m_a
            mb2 = m_b + m_c
            mt = m_a + m_b + m_c

            M = ms + mt

            prob = self._prob_three_body_escape(ms, m_b, m_c, self.q)
            # because mass attached to escaper's binary is negligible

            if np.random.rand() < prob:

                vxs, vys, vzs = np.random.normal(self.mu0, self.sig0, 3)
                vstar = np.sqrt(vxs ** 2. + vys ** 2. + vzs ** 2.)
                vxt, vyt, vzt = np.random.normal(self.mu0, self.sig0, 3)
                vbin = np.sqrt(vxt ** 2. + vyt ** 2. + vzt ** 2.)

                rdot = np.sqrt(
                    (vxs - vxt) ** 2. + (vys - vyt) ** 2. + (vzs - vzt) ** 2.)

                ebin, semi = self._sample_binding_energy(m_a, m_b, -1,
                                                         self.emin, self.emax)

            # --------------------new e0 calculation ---------------------------
                # KE_1
                initial_energy_1 = (1/2)*(ms + m_a)*rdot**2

                # PE_1
                potential_energy_1 = - grav * (m_b + m_b) / 2*semi

                # E_B1
                binary_1_energy = initial_energy_1 - (grav*(ms + m_a) / semi)

                # E_B2
                binary_2_energy = 1/2 * (m_b + m_c)*rdot**2 \
                                  - (grav * (m_b + m_c) / semi)

                e0 = initial_energy_1 + potential_energy_1 + binary_1_energy + \
                     binary_2_energy

                # e0 = 0.5 * (mb * ms / M) * (
                #         rdot ** 2.) - grav * ms * mb / self.rsep + ebin

            # ------------------------------------------------------------------

                vs = self._sample_escape_velocity_four_body(e0, ms, mt, int(npeak))

                # --------------------------------------------------------------
                # this code samples an escaper and a binary
                # Ritik will be sampling two binaries

                if vs > self.vesc0:

                    self.vesc = np.append(self.vesc, vs)

                    vxkick[nescape] = vs * (vxs / vstar)
                    vykick[nescape] = vs * (vys / vstar)
                    vzkick[nescape] = vs * (vzs / vstar)

                    if triplets:
                        # Check to see if recoil triplet will also escape
                        # triplet kick velocity is calculated assuming total
                        # linear momentum of system sums to zero

                        pxi = ms * vxs + mt * vxt
                        pyi = ms * vys + mt * vyt
                        pzi = ms * vzs + mt * vzt

                        vxkickt[nescape] = (pxi - ms * vxkick[nescape]) / mt
                        vykickt[nescape] = (pyi - ms * vykick[nescape]) / mt
                        vzkickt[nescape] = (pzi - ms * vzkick[nescape]) / mt

                        vst = np.sqrt(
                            vxkickt[nescape] ** 2. + vykickt[nescape] ** 2. +
                            vzkickt[nescape] ** 2.)

                        self.vescb = np.append(self.vescb, vst)

                        if vst > self.vesc0:
                            self.bindx[nescape] = True

                    self.mstar[nescape] = ms
                    self.mb1[nescape] = m_a
                    self.mb2[nescape] = m_b
                    self.eb[nescape] = ebin

                    nescape += 1

                if verbose:
                    print('DEBUG: ', nescape, prob, vs, self.vesc0)

        # PyCharm says the below variables are not used anywhere
        Re0, phie0, ze0, vRe0, vTe0, vze0 = np.array([]), np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([])

        # Initial and final positions and velocities
        vxvv_i = []
        vxvv_f = []

        for i in range(0, self.nstar):
            xi, yi, zi = self.o.x(self.tesc[i] / self.to), self.o.y(
                self.tesc[i] / self.to), self.o.z(self.tesc[i] / self.to)
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

            # Integrate orbit from tesc to 0.
            os = Orbit(vxvv_i[-1], ro=self.ro, vo=self.vo,
                       solarmotion=[-11.1, 24.0, 7.25])
            ts = np.linspace(self.tesc[i] / self.to, 0., 1000)
            os.integrate(ts, self.pot)

            # Save final positions and velocities
            vxvv_f.append(
                [os.R(0.) / self.ro, os.vR(0.) / self.vo, os.vT(0.) / self.vo,
                 os.z(0.) / self.ro, os.vz(0.) / self.vo, os.phi(0.)])

        # Save initial and final positions and velocities
        # of kicked stars at t=0 in orbit objects
        self.oi = Orbit(vxvv_i, ro=self.ro, vo=self.vo,
                        solarmotion=[-11.1, 24.0, 7.25])
        self.of = Orbit(vxvv_f, ro=self.ro, vo=self.vo,
                        solarmotion=[-11.1, 24.0, 7.25])

        if triplets:
            # Integrate orbits of binary star with kick velocities
            # greater than the cluster's escape speed:
            # Initial and final positions and velocities
            vxvvb_i = []
            vxvvb_f = []

            for i in range(0, self.nstar):
                xi, yi, zi = self.o.x(self.tesc[i] / self.to), self.o.y(
                    self.tesc[i] / self.to), self.o.z(self.tesc[i] / self.to)
                vxi = vxkickt[i] + self.o.vx(self.tesc[i] / self.to)
                vyi = vykickt[i] + self.o.vy(self.tesc[i] / self.to)
                vzi = vzkickt[i] + self.o.vz(self.tesc[i] / self.to)

                if verbose:
                    print('Triplet ', i, self.tesc[i], xi, yi, zi, vxi, vyi, vzi)

                # Save initial positions and velocities
                Ri, phii, zi = coords.rect_to_cyl(xi, yi, zi)
                vRi, vTi, vzi = coords.rect_to_cyl_vec(vxi, vyi, vzi, xi, yi,
                                                       zi)

                vxvvb_i.append(
                    [Ri / self.ro, vRi / self.vo, vTi / self.vo, zi / self.ro,
                     vzi / self.vo, phii])

                # Integrate orbit from tesc to 0. if kick velocity is
                # higher than cluster's escape velocity

                if self.bindx[i]:
                    os = Orbit(vxvvb_i[-1], ro=self.ro, vo=self.vo,
                               solarmotion=[-11.1, 24.0, 7.25])
                    ts = np.linspace(self.tesc[i] / self.to, 0., 1000)
                    os.integrate(ts, self.pot)
                    vxvvb_f.append([os.R(0.) / self.ro, os.vR(0.) / self.vo,
                                    os.vT(0.) / self.vo, os.z(0.) / self.ro,
                                    os.vz(0.) / self.vo, os.phi(0.)])

                else:
                    vxvvb_f.append(
                        [self.o.R(0.) / self.ro, self.o.vR(0.) / self.vo,
                         self.o.vT(0.) / self.vo, self.o.z(0.) / self.ro,
                         self.o.vz(0.) / self.vo, self.o.phi(0.)])

            self.obi = Orbit(vxvvb_i, ro=self.ro, vo=self.vo,
                             solarmotion=[-11.1, 24.0, 7.25])
            self.obf = Orbit(vxvvb_f, ro=self.ro, vo=self.vo,
                             solarmotion=[-11.1, 24.0, 7.25])

        if triplets:
            return self.of, self.obf
        else:
            return self.of

    # --------------------------------------------------------------------------

    # probability of a three body escaper
    def _prob_three_body_escape(self, ms, m_a, m_b, q):
        # using same function for 4 body because 4 body system post interaction
        # is essentially reduced to 2 body
        # (since third remaining star is negligible)

        # Equation 7.23 (eqn 1 from 3-body paper)
        prob = (ms ** q) / (ms ** q + m_a ** q + m_b ** q)
        return prob

    def _sample_binding_energy(self, mb1, mb2, alpha, emin, emax):
        # Opik's Law
        # Default binding energy distribution is:
        # power law of slope -1
        # Between 10.0**36 and 10.0**40 J

        grav = 4.302e-3  # pc/Msun (km/s)^2

        if isinstance(mb1, float):
            n = 1
        else:
            n = len(mb1)

        ebin_si = self._power_law_distribution_function(n, alpha, emin,
                                                        emax) # Joules = kg (m/s)^2
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



