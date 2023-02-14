def sample_four_body(self, tdisrupt=1000., rate=1., nstar=None, mu0=0.,
                     sig0=10.0, vesc0=10.0, rho0=1., mmin=0.1, mmax=1.4,
                     alpha=-1.35, masses=None, m1a=None, m1b=None,
                     m2a=None, m2b=None, emin=None, emax=None, balpha=-1, q=-3,
                     npeak=5., binaries=False, verbose=False, **kwargs):
    """ A function for sampling the three-body interaction core ejection
        distribution function

    Parameters - code 4-body myself in new notebook
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

------ 4 body parameters -----------------------------------------------------

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

------------------------------------------------------------------------------

    emin : float
        minimum binary energy in Joules (default: None)
    emax : float
        maximum binary energy in Joules (default: None)
    balpha : float
        power-law slope of initial binary binding energy distribution
        (default: -1)
    (need to change) q: float
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
        Sample separation between binaries within core
        (default: False)
    nrsep : float
        Numer of mean separations to sample out to when sampling separation
        between binaries (default : 2)
    initialize : bool
        initialize orbits only, do not integrate (default:False)
        Note if initialize == True then the initial, un-integrated orbits,
        will be returned

    Returns (we want everything [2+2 and 3+1])
    ----------
    of : orbit
        galpy orbit instance for kicked stars

    if binaries:
        obf : orbit
            galpy orbit instance for recoil binary stars

    History
    -------
    2021 - Written - Grandin/Webb (UofT)
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

    if binaries # always return binaries (delete all of the else clauses)
        self.binaries = True
        self.bindx = np.zeros(self.nstar, dtype=bool)
        self.vescb = np.array([])
    else: # delete
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
                        rdot ** 2.) - grav * ms * mb / self.rsep + ebin
                dr = self.rsep

            vs = self._sample_escape_velocity(e0, ms, mb, npeak, nrandom) # new function

            if vs > self.vesc0: # refactor for first binary

                self.vesc = np.append(self.vesc, vs)
                self.dr = np.append(self.dr, dr)
                vxkick[nescape] = vs * (vxs / vstar)
                vykick[nescape] = vs * (vys / vstar)
                vzkick[nescape] = vs * (vzs / vstar)

                if binaries: # refactor for second binary
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
