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


# load in parameters for
gcname = 'M92'  # GC name
mu0 = 0.  # Average 1D velocity in the core [km/s]
sig0 = 9.1  # Central 1D velocity dispersion [km/s]
vesc0 = 36.4  # Central escape velocity [km/s]
logrho0 = 6.3  # Log of central density [Msol / pc^3]
rho0 = 10.0**6.3  # Core density [Msol / pc^3]
mgc = 2.73e5  # Mass [solar masses]
rgc = 113  # Tidal radius of GC, assuming King potential [pc]
rcore = 0.29  # Core radius of GC [pc]
W0 = 9.410640945306687

mmin = 0.2  # Minimum stellar mass in core [Msol]
mmax = 0.8  # Maximum stellar mass in the core [Msol]
alpha = -0.83  # Stellar mass function in the core slope (Salpeter 1955)

potential = MWPotential2014  # Galactic potential model
ro = 8.  # radius of the solar circle (needed for galpy)
vo = 220.  # circular orbit velocity at the solar circle (needed for galpy)

# initialize cluster orbit in galpy
os_init = Orbit.from_name(gcname, ro=ro, vo=vo, solarmotion=[-11.1, 24.0, 7.25])

# integrating cluster orbit
ts = np.linspace(0, 10, 1000)  # integration times
os_init.integrate(ts, potential)  \
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
os, ob = cspray.sample_three_body(p_orb, nstar=10000, mu0=mu0,
                                  sig0=sig0, vesc0=vesc0,
                                  rho0=rho0, binaries=True)

# -----------------------------------------------------------------------------
# Sample for one orbital period (4 body)
os_four, ot = cspray_fourbody.sample_four_body(p_orb, nstar=10000, mu0=mu0,
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

# bindx is a boolean that gives indxs of binaries that escaped the cluster:
bin_esc = cspray.bindx
print("{}/{} binaries escaped {}.".format(len(ra_b[bin_esc]), len(ra_b),
                                          gcname))


# -----------------------------------------------------------------------------
# tindx is a boolean that gives indxs of triples that escaped the cluster:
tin_esc = cspray_fourbody.tindx
print("{}/{} triples escaped {}.".format(len(ra_t[tin_esc]), len(ra_t),
                                         gcname))
# -----------------------------------------------------------------------------


# ---------------------  plots for 3 body simulations -------------------------
# explore parameter spaces of simulations
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# RA vs Dec positions:
ax[0, 0].scatter(ra, dec, marker='o', s=50, edgecolor='black', c='#8d99ae',
                 alpha=0.9)
ax[0, 0].scatter(ra_b[bin_esc], dec_b[bin_esc], marker='o', s=50,
                 edgecolor='black', c='#ef233c', alpha=0.9)
ax[0, 0].scatter(cspray.o.ra(), cspray.o.dec(), marker='*', s=250,
                 edgecolor='yellow', c='#2b2d42')
# ------------------------ four body -------------------------------------------
ax[0, 0].scatter(ra_four, dec_four, marker='^', s=50, edgecolor='black', c='#86eb91',
                  alpha=0.9)
ax[0, 0].scatter(ra_t[tin_esc], dec_t[tin_esc], marker='^', s=50,
                  edgecolor='black', c='#2d30e3', alpha=0.9)
# ax[0, 0].scatter(cspray_fourbody.o.ra(), cspray_fourbody.o.dec(),
#                   marker='*', s=250, edgecolor='black', c='#2b2d42')
# -----------------------------------------------------------------------------
ax[0, 0].set_xlabel(r"$\alpha$ ($^\circ $)", fontsize=18)
ax[0, 0].set_ylabel(r"$\delta$ ($^\circ $)", fontsize=18)
ax[0, 0].tick_params(axis='both', which='major', labelsize=16)
ax[0, 0].tick_params(axis='both', which='minor', labelsize=16)

# R vs z positions:
ax[0, 1].scatter(r, z, marker='o', s=50, edgecolor='black', c='#8d99ae',
                 alpha=0.9, label='(3) Extra-tidal stars')
ax[0, 1].scatter(r_b[bin_esc], z_b[bin_esc], marker='o', s=50,
                 edgecolor='black', c='#ef233c', alpha=0.9,
                 label='Extra-tidal binaries')
ax[0, 1].scatter(cspray.o.r(), cspray.o.z(), marker='*', s=250,
                 edgecolor='yellow', c='#2b2d42', label='M92 Centre')
# ------------------------- four body ------------------------------------------
ax[0, 1].scatter(r_four, z_four, marker='^', s=50, edgecolor='black', c='#86eb91',
                  alpha=0.9, label='(4) Extra-tidal stars')
ax[0, 1].scatter(r_t[tin_esc], z_t[tin_esc], marker='^', s=50,
                  edgecolor='black', c='#2d30e3', alpha=0.9,
                  label='Extra-tidal triples')
# ax[0, 1].scatter(cspray_fourbody.o.r(), cspray_fourbody.o.z(), marker='*',
#                   s=250, edgecolor='black', c='#2b2d42', label='M3 Centre')
# ------------------------------------------------------------------------------
ax[0, 1].set_xlabel(r"$r$ (kpc)", fontsize=18)
ax[0, 1].set_ylabel(r"$z$ (kpc)", fontsize=18)
ax[0, 1].tick_params(axis='both', which='major', labelsize=16)
ax[0, 1].tick_params(axis='both', which='minor', labelsize=16)
ax[0, 1].legend(fontsize=10)

# Proper motion:
ax[1, 0].scatter(pmra, pmdec, marker='o', s=50, edgecolor='black', c='#8d99ae',
                 alpha=0.9)
ax[1, 0].scatter(pmra_b[bin_esc], pmdec_b[bin_esc], marker='o', s=50,
                 edgecolor='black', c='#ef233c', alpha=0.9)
ax[1, 0].scatter(cspray.o.pmra(), cspray.o.pmdec(), marker='*', s=250,
                 edgecolor='yellow', c='#2b2d42')
# ------------------------ four body ------------------------------------------
ax[1, 0].scatter(pmra_four, pmdec_four, marker='^', s=50, edgecolor='black',
                  c='#86eb91', alpha=0.9)
ax[1, 0].scatter(pmra_t[tin_esc], pmdec_t[tin_esc], marker='^', s=50,
                  edgecolor='black', c='#2d30e3', alpha=0.9)
# ax[1, 0].scatter(cspray_fourbody.o.pmra(), cspray_fourbody.o.pmdec(),
#                   marker='*', s=250, edgecolor='black', c='#2b2d42')
# -----------------------------------------------------------------------------
ax[1, 0].set_xlabel(r"$\mu_{\alpha}$ (mas/yr)", fontsize=18)
ax[1, 0].set_ylabel(r"$\mu_{\delta}$ (mas/yr)", fontsize=18)
ax[1, 0].tick_params(axis='both', which='major', labelsize=16)
ax[1, 0].tick_params(axis='both', which='minor', labelsize=16)

# Escape velocities:
ax[1, 1].hist(v_esc, bins=7, color='white', edgecolor='#bc5ed6', density=True, label='3body escaper')
ax[1, 1].hist(v_esc_b, bins=5, color='white', edgecolor='#ef233c', density=True, label='3body binaries')
ax[1, 1].axvline(vesc0, ls='--', c='#2b2d42')
# -------------------------- four body ----------------------------------------
ax[1, 1].hist(v_esc_four, bins=7, color='white', edgecolor='#689c78', density=True, label='4body escaper')
ax[1, 1].hist(v_esc_t, bins=5, color='white', edgecolor='#2035f5', density=True, label='4body triples')
# ax[1, 1].axvline(vesc0, ls='dashdot', c='#2b2d42', label='4body')
# -----------------------------------------------------------------------------
ax[1, 1].tick_params(axis='both', which='major', labelsize=16)
ax[1, 1].tick_params(axis='both', which='minor', labelsize=16)
ax[1, 1].set_xlabel(r"$v_{esc}$ (km/s)", fontsize=18)
ax[1, 1].set_ylabel("Normalized Counts", fontsize=18)
ax[1, 1].legend(fontsize=10)

fig.tight_layout()
plt.savefig("Ritik's M92 plot (3 and 4 bodies)(10000 stars).png")
plt.show()





