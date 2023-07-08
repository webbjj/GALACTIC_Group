# imports
# 3 body import
from corespray import corespraydf
# 4 body import
from corespraydf_4body import corespraydf_4body

# Import galpy packages:

from galpy.potential import MWPotential2014
from galpy.potential import KingPotential
from galpy.util import conversion
from galpy.orbit import Orbit

# Import other necessary Python packages:
import numpy as np
import astropy
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import animation
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


# load in parameters for M3
gcname = 'M3'  # GC name
mu0 = 0.  # Average 1D velocity in the core [km/s]
sig0 = 7.6  # Central 1D velocity dispersion [km/s]
vesc0 = 30.0  # Central escape velocity [km/s]
logrho0 = 3.67  # Log of central density [Msol / pc^3]
rho0 = 10.0**3.67  # Core density [Msol / pc^3]
mgc = 4.06e5  # Mass [solar masses]
rgc = 127.28  # Tidal radius of GC, assuming King potential [pc]
rcore = 1.23  # Core radius of GC [pc]
W0 = 8.61469762517307

mmin = 0.1  # Minimum stellar mass in core [Msol]
mmax = 1.4  # Maximum stellar mass in the core [Msol]
alpha = -1.35  # Stellar mass function in the core slope (Salpeter 1955)

potential = MWPotential2014  # Galactic potential model
ro = 8.  # radius of the solar circle (needed for galpy)
vo = 220.  # circular orbit velocity at the solar circle (needed for galpy)

# initialize cluster orbit in galpy
os_init = Orbit.from_name(gcname, ro=ro, vo=vo, solarmotion=[-11.1, 24.0, 7.25])

# integrating cluster orbit
ts = np.linspace(0, 10, 1000)  # integration times
os_init.integrate(ts, potential) \
    # can choose which Galactic potential to use (e.g. MWPotential2014)

# find maximum orbital period for cluster
p_orb_rad = os_init.Tr()
p_orb_phi = os_init.Tp()
p_orb_max = max(np.array([p_orb_rad, p_orb_phi]))
print("The maximum orbital period of {} is {} Gyr.".format(gcname, p_orb_max))

# Initialize 3 body corespray
cspray = corespraydf(gcname, potential, mgc, rgc, W0)

# -----------------------------------------------------------------------------
# Initialize 4 body corespray
cspray_fourbody = corespraydf_4body(gcname, potential, mgc, rgc, W0)
# -----------------------------------------------------------------------------

# Sample for one orbital period (3 body)
p_orb = p_orb_max * 1000  # Myr
os, ob = cspray.sample_three_body(p_orb, nstar=1000, mu0=mu0,
                                  sig0=sig0, vesc0=vesc0,
                                  rho0=rho0, binaries=True)

# -----------------------------------------------------------------------------
# Sample for one orbital period (4 body)
os_four, ot = cspray_fourbody.sample_four_body(p_orb, nstar=1000, mu0=mu0,
                                               sig0=sig0, vesc0=vesc0,
                                               rho0=rho0, triples=True)
# -----------------------------------------------------------------------------

# extract parameters for single escapers (3 body)
ra = os.ra()  # [deg]
dec = os.dec()  # [deg]
dist = os.dist()  # [kpc]
pmra = os.pmra()  # [mas/yr]
pmdec = os.pmdec()  # [mas/yr]
vr = os.vlos()  # [km/s]
r = os.R()  # [kpc]
z = os.z()  # [kpc]
t_esc = cspray.tesc  # [Myr] in the past
v_esc = cspray.vesc  # [km/s]
mstar = cspray.mstar  # [Msol]

# -----------------------------------------------------------------------------
# extract parameters for single escapers (4 body)
ra_four = os_four.ra()  # [deg]
dec_four = os_four.dec()  # [deg]
dist_four = os_four.dist()  # [kpc]
pmra_four = os_four.pmra()  # [mas/yr]
pmdec_four = os_four.pmdec()  # [mas/yr]
vr_four = os_four.vlos()  # [km/s]
r_four = os_four.R()  # [kpc]
z_four = os_four.z()  # [kpc]
t_esc_four = cspray_fourbody.tesc  # [Myr] in the past
v_esc_four = cspray_fourbody.vesc  # [km/s]
mstar_four = cspray_fourbody.mstar  # [Msol]
# -----------------------------------------------------------------------------

# extract parameters for all binary stars
ra_b = ob.ra()  # [deg]
dec_b = ob.dec()  # [deg]
dist_b = ob.dist()  # [kpc]
pmra_b = ob.pmra()  # [mas/yr]
pmdec_b = ob.pmdec()  # [mas/yr]
vr_b = ob.vlos()  # [km/s]
r_b = ob.R()  # [kpc]
z_b = ob.z()  # [kpc]
# Would tesc of binary be same as escaper star?
v_esc_b = cspray.vescb  # [km/s]
# print("vesc_b is:", v_esc_b)
# print("vesc is:", v_esc)
m_b1 = cspray.mb1  # [Msol]
m_b2 = cspray.mb2  # [Msol]
m_b_tot = m_b1 + m_b2  # [Msol]

# -----------------------------------------------------------------------------
# extract parameters for all triple stars
ra_t = ot.ra()  # [deg]
dec_t = ot.dec()  # [deg]
dist_t = ot.dist()  # [kpc]
pmra_t = ot.pmra()  # [mas/yr]
pmdec_t = ot.pmdec()  # [mas/yr]
vr_t = ot.vlos()  # [km/s]
r_t = ot.R()  # [kpc]
z_t = ot.z()  # [kpc]
# Would tesc of triple be same as escaper star?
v_esc_t = cspray_fourbody.vesct  # [km/s]
# print("vesc_t is:", v_esc_t)
# print("vesc is:", v_esc_four)
m_b1_four = cspray_fourbody.mb1  # [Msol]
m_b2_four = cspray_fourbody.mb2  # [Msol]
m_b_tot_four = m_b1_four + m_b2_four  # [Msol]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# extract parameters for three body energies
eb_three = cspray.eb
eb_0 = cspray.e0
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# extract parameters for four body energies
et_1 = cspray_fourbody.eb1
et_2 = cspray_fourbody.eb2
et_0 = cspray_fourbody.e0

# bindx is a boolean that gives indxs of binaries that escaped the cluster:
bin_esc = cspray.bindx
print("{}/{} binaries escaped {}.".format(len(ra_b[bin_esc]), len(ra_b),
                                          gcname))


# -----------------------------------------------------------------------------
# tindx is a boolean that gives indxs of triples that escaped the cluster:
tin_esc = cspray_fourbody.tindx
print("{}/{} triples escaped {}.".format(len(ra_t[tin_esc]), len(ra_t),
                                         gcname))


# histogram of 3body escaper mass distribution and
# 4body escaper mass distribution
plt.figure(1)
_, bin_num, _ = plt.hist(mstar, bins=20, color='red', edgecolor='black',
                         alpha=0.4, density=False, label='3body escaper mass dist.')
_ = plt.hist(mstar_four, bins=bin_num, color='blue', edgecolor='black',
             alpha=0.4, density=False, label='4body escaper mass dist.')
plt.xlabel("Masses", fontsize=13)
plt.ylabel("Counts", fontsize=13)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('Comparsion of escaper masses (1000 stars, M3).png')
plt.show()

# histogram of 3body binary distribution and 4body triple distribution
plt.figure(2)
_, bin_num, _ = plt.hist(m_b_tot, bins=20, color='red', edgecolor='black',
                         alpha=0.4, density=True, label='3body binary mass dist.')
_ = plt.hist(m_b_tot_four, bins=bin_num, color='blue', edgecolor='black',
             alpha=0.4, density=True, label='4body triple mass dist.')
plt.xlabel("Masses", fontsize=13)
plt.ylabel("Normalized Counts", fontsize=13)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('Comparsion of binary and triple masses (1000 stars, M3).png')
plt.show()


# histogram of 3body single escape vel. distribution and
# 4body single escape vel. distribution
plt.figure(3)
_, bin_num, _ = plt.hist(v_esc, bins=20, color='red', edgecolor='black',
                         alpha=0.4, density=True, label='3body single esc. vel. dist.')
_ = plt.hist(v_esc_four, bins=bin_num, color='blue', edgecolor='black',
             alpha=0.4, density=True, label='4body single esc. vel. dist.')
plt.axvline(vesc0, ls='--', c='black')
plt.xlabel(r"$v_{esc}$ (km/s)", fontsize=13)
plt.ylabel("Normalized Counts", fontsize=13)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('Comparison of single\'s escape velocity (1000 stars, M3).png')
plt.show()


# histogram of 3body binary escape vel. distribution and
# 4body triple escape vel. distribution
plt.figure(4)
_, bin_num, _ = plt.hist(v_esc_b, bins=20, color='red', edgecolor='black',
                         alpha=0.4, density=True, label='3body binary esc. vel. dist.')
_ = plt.hist(v_esc_t, bins=bin_num, color='blue', edgecolor='black',
             alpha=0.4, density=True, label='4body triple esc. vel. dist.')
plt.axvline(vesc0, ls='--', c='black')
plt.xlabel(r"$v_{esc}$ (km/s)", fontsize=13)
plt.ylabel("Normalized Counts", fontsize=13)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('Comparison of binaries and triples escape velocities (1000 stars, M3).png')
plt.show()


# histogram of 3body initial energy distribution and
# 4body initial energy distribution
plt.figure(5)
_, bin_num, _ = plt.hist(eb_0, bins=20, color='red', edgecolor='black',
                         alpha=0.4, density=True, label='3body initial energy dist.')
_ = plt.hist(et_0, bins=bin_num, color='blue', edgecolor='black',
             alpha=0.4, density=True, label='4body initial energy dist.')
plt.axvline(vesc0, ls='--', c='black')
plt.xlabel("Energies", fontsize=13)
plt.ylabel("Normalized Counts", fontsize=13)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('Comparison of 3 and 4 body initial energies (1000 stars, M3).png')
plt.show()


