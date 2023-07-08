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
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# RA vs Dec positions:
# ax[0, 0].scatter(ra, dec, marker='o', s=50, edgecolor='black', c='#8d99ae',
#                  alpha=0.9)
# ax[0, 0].scatter(ra_b[bin_esc], dec_b[bin_esc], marker='o', s=50,
#                  edgecolor='black', c='#ef233c', alpha=0.9)
# ax[0, 0].scatter(cspray.o.ra(), cspray.o.dec(), marker='*', s=250,
#                  edgecolor='yellow', c='#2b2d42')
# ------------------------ four body -------------------------------------------
plt.figure(1)
plt.scatter(ra, dec, marker='o', s=50, edgecolor='black', c='#8d99ae', alpha=0.4,
            label='3body escapers')

plt.scatter(ra_four, dec_four, marker='^', s=50, edgecolor='black', c='#86eb91',
            alpha=0.4, label='4body escapers')

plt.scatter(ra_b[bin_esc], dec_b[bin_esc], marker='o', s=50, edgecolor='black',
            c='#ef233c', alpha=0.4, label='3body binaries')

plt.scatter(ra_t[tin_esc], dec_t[tin_esc], marker='^',
            edgecolor='black', c='#2d30e3', alpha=0.4, label='4body triples')
plt.scatter(cspray_fourbody.o.ra(), cspray_fourbody.o.dec(),
            marker='*', s=250, edgecolor='yellow', c='black', label='M3 Centre')
# -----------------------------------------------------------------------------
plt.xlabel(r"$\alpha$ ($^\circ $)", fontsize=15)
plt.ylabel(r"$\delta$ ($^\circ $)", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tick_params(axis='both', which='minor', labelsize=13)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("(Ra vs Dec)(3 and 4 bodies)(10,000 stars).png")
plt.show()

# R vs z positions:
# ax[0, 1].scatter(r, z, marker='o', s=50, edgecolor='black', c='#8d99ae',
#                  alpha=0.9, label='(3) Extra-tidal stars')
# ax[0, 1].scatter(r_b[bin_esc], z_b[bin_esc], marker='o', s=50,
#                  edgecolor='black', c='#ef233c', alpha=0.9,
#                  label='Extra-tidal binaries')
# ax[0, 1].scatter(cspray.o.r(), cspray.o.z(), marker='*', s=250,
#                  edgecolor='yellow', c='#2b2d42', label='M3 Centre')
# ------------------------- four body ------------------------------------------
plt.figure(2)
plt.scatter(r, z, marker='o', s=50, edgecolor='black', c='#8d99ae',
            alpha=0.4, label='(3) Extra-tidal stars')

plt.scatter(r_four, z_four, marker='^', s=50, edgecolor='black', c='#86eb91',
            alpha=0.4, label='(4) Extra-tidal stars')

plt.scatter(r_b[bin_esc], z_b[bin_esc], marker='o', s=50, edgecolor='black',
            c='#ef233c', alpha=0.4, label='Extra-tidal binaries')

plt.scatter(r_t[tin_esc], z_t[tin_esc], marker='^', s=50, edgecolor='black',
            c='#2d30e3', alpha=0.4, label='Extra-tidal triples')
plt.scatter(cspray_fourbody.o.r(), cspray_fourbody.o.z(), marker='*',
            s=250, edgecolor='yellow', c='black', label='M3 Centre')
# ------------------------------------------------------------------------------
plt.xlabel(r"$r$ (kpc)", fontsize=15)
plt.ylabel(r"$z$ (kpc)", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tick_params(axis='both', which='minor', labelsize=13)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("(R vs z)(3 and 4 bodies)(10,000 stars).png")
plt.show()

# Proper motion:
# ax[1, 0].scatter(pmra, pmdec, marker='o', s=50, edgecolor='black', c='#8d99ae',
#                  alpha=0.9)
# ax[1, 0].scatter(pmra_b[bin_esc], pmdec_b[bin_esc], marker='o', s=50,
#                  edgecolor='black', c='#ef233c', alpha=0.9)
# ax[1, 0].scatter(cspray.o.pmra(), cspray.o.pmdec(), marker='*', s=250,
#                  edgecolor='yellow', c='#2b2d42')
# ------------------------ four body ------------------------------------------
plt.figure(3)
plt.scatter(pmra, pmdec, marker='o', s=50, edgecolor='black', c='#8d99ae',
            alpha=0.4, label='(3) Extra-tidal stars')

plt.scatter(pmra_four, pmdec_four, marker='^', s=50, edgecolor='black',
            c='#86eb91', alpha=0.4, label='(4) Extra-tidal stars')

plt.scatter(pmra_b[bin_esc], pmdec_b[bin_esc], marker='o', s=50,
            edgecolor='black', c='#ef233c', alpha=0.4, label='Extra-tidal binaries')

plt.scatter(pmra_t[tin_esc], pmdec_t[tin_esc], marker='^', s=50,
            edgecolor='black', c='#2d30e3', alpha=0.4, label='Extra-tidal triples')
plt.scatter(cspray_fourbody.o.pmra(), cspray_fourbody.o.pmdec(),
            marker='*', s=250, edgecolor='yellow', c='black', label='M3 Centre')
# -----------------------------------------------------------------------------
plt.xlabel(r"$\mu_{\alpha}$ (mas/yr)", fontsize=15)
plt.ylabel(r"$\mu_{\delta}$ (mas/yr)", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tick_params(axis='both', which='minor', labelsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("(PMRa vs PMDec)(3 and 4 bodies)(10,000 stars).png")
plt.show()

# Escape velocities:
# ax[1, 1].hist(v_esc, bins=7, color='white', edgecolor='#bc5ed6', density=True, label='3body escaper')
# ax[1, 1].hist(v_esc_b, bins=5, color='white', edgecolor='#ef233c', density=True, label='3body binaries')
# ax[1, 1].axvline(vesc0, ls='--', c='#2b2d42')
# -------------------------- four body ----------------------------------------
plt.figure(4)
_, bin_num, _ = plt.hist(v_esc, bins=20, color='yellow', edgecolor='black', alpha=0.3,
                         density=True, label='3body escaper')
_ = plt.hist(v_esc_b, bins=bin_num, color='red', edgecolor='black', alpha=0.3,
             density=True, label='3body binaries')
_ = plt.hist(v_esc_four, bins=bin_num, color='green', edgecolor='black', alpha=0.3,
             density=True, label='4body escaper')
_ = plt.hist(v_esc_t, bins=bin_num, color='blue', edgecolor='black', alpha=0.3,
             density=True, label='4body triples')
plt.axvline(vesc0, ls='--', c='#2b2d42')
# -----------------------------------------------------------------------------
plt.xlim(0, 300)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tick_params(axis='both', which='minor', labelsize=13)
plt.xlabel(r"$v_{esc}$ (km/s)", fontsize=15)
plt.ylabel("Normalized Counts", fontsize=15)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("(esc. vel.)(3 and 4 bodies)(10,000 stars).png")
plt.show()

# ------------------------- zoomed in figure 4 ---------------------------------
plt.figure(5)
_, bin_num, _ = plt.hist(v_esc, bins=20, color='yellow', edgecolor='black', alpha=0.3,
                         density=True, label='3body escaper')
_ = plt.hist(v_esc_b, bins=bin_num, color='red', edgecolor='black', alpha=0.3,
             density=True, label='3body binaries')
_ = plt.hist(v_esc_four, bins=bin_num, color='green', edgecolor='black', alpha=0.3,
             density=True, label='4body escaper')
_ = plt.hist(v_esc_t, bins=bin_num, color='blue', edgecolor='black', alpha=0.3,
             density=True, label='4body triples')
plt.axvline(vesc0, ls='--', c='#2b2d42')
# -----------------------------------------------------------------------------
plt.xlim(175, 400)
plt.ylim(0.0, 0.0015)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tick_params(axis='both', which='minor', labelsize=13)
plt.xlabel(r"$v_{esc}$ (km/s)", fontsize=15)
plt.ylabel("Normalized Counts", fontsize=15)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("(esc. vel.)(3 and 4 bodies)(10,000 stars)(zoomed).png")
plt.show()

# fig.tight_layout()
# plt.savefig("Ritik's M3 plot (4 bodies)(10,000 stars).png")
# plt.show()

# ------------------------- distribution of Ra and Dec -------------------------
ra_m3 = 205.548416
dec_m3 = 28.377277

dra_three_sing = (ra - ra_m3) * np.cos(np.radians(dec_m3))
ddec_three_sing = dec - dec_m3
dr_three_sing = np.sqrt(dra_three_sing**2. + ddec_three_sing**2.)

dra_three_bin = (ra_b - ra_m3) * np.cos(np.radians(dec_m3))
ddec_three_bin = dec_b - dec_m3
dr_three_bin = np.sqrt(dra_three_bin**2. + ddec_three_bin**2.)

dra_four_sing = (ra_four - ra_m3) * np.cos(np.radians(dec_m3))
ddec_four_sing = dec_four - dec_m3
dr_four_sing = np.sqrt(dra_four_sing**2. + ddec_four_sing**2.)

dra_four_trip = (ra_t - ra_m3) * np.cos(np.radians(dec_m3))
ddec_four_trip = dec_t - dec_m3
dr_four_trip = np.sqrt(dra_four_trip**2. + ddec_four_trip**2.)

plt.figure(6)
_, bin_num, _ = plt.hist(dr_three_sing, bins=20, color='yellow', edgecolor='black', alpha=0.3,
                         density=True, label='3body escaper')
_ = plt.hist(dr_four_sing, bins=bin_num, color='green', edgecolor='black', alpha=0.3,
             density=True, label='4body escaper')
_ = plt.hist(dr_three_bin, bins=bin_num, color='red', edgecolor='black', alpha=0.3,
             density=True, label='3body binaries')
_ = plt.hist(dr_four_trip, bins=bin_num, color='blue', edgecolor='black', alpha=0.3,
             density=True, label='4body triples')
# plt.axvline(vesc0, ls='--', c='#2b2d42')
# -----------------------------------------------------------------------------
# plt.xlim(0, 300)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tick_params(axis='both', which='minor', labelsize=13)
plt.xlabel("Angular Sep. (deg)", fontsize=15)
plt.ylabel("Normalized Counts", fontsize=15)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("(angular sep.)(3 and 4 bodies)(10,000 stars).png")
plt.show()



