from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, PlummerPotential, KingPotential, \
    MovingObjectPotential
from galpy.util import conversion, coords
import numpy as np
import scipy as sp
from scipy import constants

import matplotlib.pyplot as plt
from matplotlib import animation

import time


# def _sample_escape_velocity_four_body(e0, ms, mb, n, vs, npeak=5,
# nrandom=1000):  # add new parameters (might need vs)

def _sample_escape_velocity_four_body(e0, ms, mb, npeak=5, nrandom=1000):
    # randomly sample between npeak * vs_peak

    vs_peak = _escape_velocity_distribution_peak_four_body(e0, ms, mb)
    match = False

    while not match:
        vstemp = np.random.rand(nrandom) * npeak * vs_peak
        amptemp = np.random.rand(nrandom) * vs_peak

        aindx = amptemp < _escape_velocity_distribution_four_body(
                    np.ones(nrandom) * vstemp, np.ones(nrandom) * e0,
                    np.ones(nrandom) * ms, np.ones(nrandom) * mb)

        if np.sum(aindx) > 0:
            vs = vstemp[aindx][0]
            match = True

    return vs

# plot curve of sample escape velocities and sample


def _escape_velocity_distribution_four_body(vs, e0, ms, mb, n=3):
    # Equation 24 from Leigh et al., 2021
    M = ms + mb
    fv = ((n - 1) * (np.fabs(e0) ** (n - 1)) * ms * M / mb) * vs / \
         ((np.fabs(e0) + 0.5 * (ms * M / mb) * (vs ** 2.)) ** n)
    return fv


def _escape_velocity_distribution_peak_four_body(e0, ms, mb, n=3):
    # Equation 25 from Leigh et al., 2021
    M = ms + mb
    epsilon = (n - 0.5) ** (-0.5)
    vs_peak = epsilon * ((M - ms) / (ms * M)) ** 0.5 * (np.fabs(e0)) ** 0.5

    return vs_peak


# m_sun = 1.989 * 10**30  # kg

g_constant = 4.301 * 10**-9  # km^2 Mpc M_Sun^-1 s^-2

# g_constant = 6.67 * 10**-11  # m^3 kg^-1 s^-2


# ----------------------- these form binary 1 --------------------------------
m_single = 1.5  # * m_sun  # Mass [multiples of solar mass]
m1 = 2  # * m_sun
# ----------------------------------------------------------------------------

# ------------------------ these form binary 2 -------------------------------
m2 = 1.8  # * m_sun
m3 = 2.2  # * m_sun
# ----------------------------------------------------------------------------

m_b_1 = m_single + m1
m_b_2 = m2 + m3

m_triplet = m1 + m2 + m3
m_total = m_triplet + m_single

relative_vel = 10  # km/s (might need to change)
semi = 1.496 * 10**9  # km  # 10 AU (might need to change)
energy_single = (1/2) * m_single * relative_vel**2  # kinetic energy in J
# energy_single = 2.9835 * 10**38  # kinetic energy in J


# -------------------- this energy is for after interaction -------------------
energy_b_a = -(g_constant * m_single * m1) / 2 * semi  # confirm units for G do first (fixed)
energy_b_b = -(g_constant * m2 * m3) / 2 * semi
energy_triplet = energy_b_a + energy_b_b
energy_total = energy_triplet + energy_single
# -----------------------------------------------------------------------------

"""
for initial energy before interaction, consider 2 binaries that interact.
find the relative separation between binary 1 and binary 2, and also the 
relative velocity 
we assume that the star with the lowest mass out of the four will be the escaper
and its partner star will be negligible after the interaction. 
Let the binary with the lowest mass star be binary 1 and the other binary be 
binary 2.
Initial energy = KE_1 + PE_1 + E_B1 + E_B2
KE_1 is kinetic energy of the first binary 
PE_1 is the potential energy of the lowest mass star's partner relative to the 
second binary
E_B1, E_B2 are the energies of the first and second binaries 
initial_energy_1 = 1/2 m v_{b1}^2 
potential_energy_1 = - G M_{heavier binary}^2 / 2R
binary_1_energy = initial_energy_1 - (G M_{star1} M_{star2} / R)
binary_2_energy = 1/2 (M_{star3} + M_{star4}) v_{b2}^2 - (G M_{star3} M_{star4} / R)
"""

# KE_1
initial_energy_1 = (1/2)*(m_single + m1)*relative_vel**2

# PE_1
potential_energy_1 = - g_constant * (m2 + m3) / 2*semi

# E_B1
binary_1_energy = initial_energy_1 - (g_constant*(m_single + m1) / semi)

# E_B2
binary_2_energy = 1/2 * (m2 + m3)*relative_vel**2 - (g_constant * (m2 + m3) /
                                                     semi)

initial_energy = initial_energy_1 + potential_energy_1 + binary_1_energy + \
                 binary_2_energy

print("Initial energy before interaction:", initial_energy)

print("energy inner:", energy_b_a)
print("energy outer:", energy_b_b)
print("energy single:", energy_single)
print("energy triplet:", energy_triplet)
print("energy total:", energy_total)


# escaper function takes in masses of each star involved in system and returns
# which star will escape based on mass
def escaper(mass1, mass2, mass3, mass4):
    return min([mass1, mass2, mass3, mass4])


# m_0 is average mass of four bodies
def m_naught(m_a, m_b, m_c, m_s):  # modified eqn 7.28 from Valtonen 2003
    return ((m_a * (m_b + m_c) + m_b * m_c + m_s * (m_a + m_b + m_c)) / 4)**0.5


# L_0 is total angular momentum, normalized by L = L_0 / L_max
# L_max is maximum angular momentum of system
# L_max = 2.5 * g_constant * (m_naught(m1, m2, m3, m_single)**5 /
#                                 np.fabs(energy_total))**0.5
#
# print("L_max", L_max)

L1 = 0  # / L_max
# L2 = 0.15 / L_max
# L3 = 0.4 / L_max


n1 = 18 * L1**2 + 3  # [n = 3] # eqn for n from Leigh et al., 2021
# n2 = 18 * L2**2 + 3
# n3 = 18 * L3**2 + 3


velocities_0 = []
i = 0
while i < 100:
    a = _sample_escape_velocity_four_body(initial_energy, m_b_1, m_b_2)
    # while a > 10:
    #     a = _sample_escape_velocity_four_body(initial_energy, m_b_1, m_b_2)
    velocities_0.append(a)
    i += 1


plt.figure(1)
plt.style.use('seaborn')
plt.hist(velocities_0, bins=20, color='blue', edgecolor='black', linewidth=1.2,
         density=False)
plt.xlabel("esc. velocity (km/s)")
plt.ylabel("count")
plt.title("Sample Escape Velocities for 4-body Function with 2 Binary Input")
plt.savefig("escvel(4body for 2 binary) new vel.png")
plt.show()


# ------------------- plot of escape velocity distribution --------------------

velocity_dis = []
for vs_value in np.arange(0, 100, 1):
    a = _escape_velocity_distribution_four_body(vs_value, initial_energy,
                                                m_b_1, m_b_2)
    velocity_dis.append(a)


plt.figure(2)
plt.style.use('seaborn')
plt.plot(np.arange(0, 100, 1), velocity_dis, color='red', label='fitted curve')
plt.bar(np.arange(0, 100, 1), velocity_dis, color='blue', edgecolor='black', linewidth=1.2)
plt.xlabel("v_s values")
plt.ylabel("Velocities")
plt.title("Escape Velocity Distribution (with Fitted Curve) for 4-body Function with 2 Binary Input")
plt.legend()
plt.savefig("escveldis_curve (4body for 2 binary).png")
plt.show()


velocities_1 = []
x_values = np.arange(0, 10, 0.1)
j = 0
while j < 100:
    a = _sample_escape_velocity_four_body(initial_energy, m_b_1, m_b_2)
    # while a > 10:
    #     a = _sample_escape_velocity_four_body(initial_energy, m_b_1, m_b_2)
    velocities_1.append(a)
    j += 1

for i in range(len(velocities_1)):
    velocities_1[i] = (velocities_1[i] - min(velocities_1)) / \
                      (max(velocities_1) - min(velocities_1))


# -----------------------------------------------------------------------------

# curve of values received from sample_escape_velocity function
# plt.figure(3)
# plt.style.use('seaborn')
# plt.plot(x_values, velocities_1, color='blue', linestyle='-')
# plt.xlabel("esc. velocity (km/s)")
# plt.ylabel("normalized count")
# plt.title("Curve of Sample Escape Velocities for 4-body Function with 2 Binary Input")
# plt.savefig("curve of escvel(4body for 2 binary).png")
# plt.show()

# ------------------------------------------------------------------------------

# velocities_015 = []
# j = 0
# while j < 10:
#     a = _sample_escape_velocity_four_body(energy_total, m_single, m_triplet,
#                                           n2)
#     velocities_015.append(a)
#     j += 1
# vs_four = _sample_escape_velocity_four_body(energy_total, m_single, m_triplet,
#                                             n1)

# plt.figure(2)
# plt.hist(velocities_015, bins=7, color='#8d99ae', density=True)
# plt.xlabel("esc. velocity (km/s)")
# plt.ylabel("normalized count")
# plt.title("Sample Escape Velocities for 4-body Function with L_0 = 0.15")
# plt.savefig("escvel(4body for L_0 = 0.15).png")
# plt.show()
#
# velocities_04 = []
# k = 0
# while k < 10:
#     a = _sample_escape_velocity_four_body(energy_total, m_single, m_triplet,
#                                           n3)
#     velocities_04.append(a)
#     k += 1
# # vs_four = _sample_escape_velocity_four_body(energy_total, m_single, m_triplet,
# #                                             n1)
#
# plt.figure(3)
# plt.hist(velocities_04, bins=7, color='#8d99ae', density=True)
# plt.xlabel("esc. velocity (km/s)")
# plt.ylabel("normalized count")
# plt.title("Sample Escape Velocities for 4-body Function with L_0 = 0.4")
# plt.savefig("escvel(4body for L_0 = 0.4).png")
# plt.show()


# ------- BELOW CODE IS NO LONGER NECESSARY BECAUSE n = 3 IS FIXED NOW ---------

# create distribution of sample velocities peaks with n values from 0 to 100
# -----------------------------------------------------------------------------
# peaks = []
# for n in range(0, 101):
#     peaks.append(_escape_velocity_distribution_peak_four_body(initial_energy,
#                                                               m_b_1,
#                                                               m_b_2))
#
#
# x = np.arange(0, 101, 1)
#
# print(peaks)
#
# plt.figure(3)
# plt.style.use('seaborn')
# plt.plot(x, peaks, color='black', linestyle='dashed')
# plt.plot([x[1]], [peaks[1]], marker="*", markersize=12, color="purple",
#          label='n=1')
# plt.plot([x[3]], [peaks[3]], marker="*", markersize=12, color="orange",
#          label='n=3')
# plt.scatter(x, peaks, marker='o', s=15, color='red')
# plt.title("Distribution of Escape Velocity Peaks for n = [0, 100]")
# plt.xlabel("n")
# plt.ylabel("value of peak in km/s")
# plt.legend()
# plt.savefig("distribution 4-esc vel.png")
# plt.show()
#
#
# plt.figure(4)
# plt.style.use('seaborn')
# plt.plot(x, peaks, color='black', linestyle='dashed')
# plt.scatter(x, peaks, marker='o', s=15, color='red')
# plt.plot([x[1]], [peaks[1]], marker="*", markersize=12, color="purple",
#          label='n=1')
# plt.plot([x[3]], [peaks[3]], marker="*", markersize=12, color="orange",
#          label='n=3')
# plt.title("Distribution of Escape Velocity Peaks for n = [0, 100] (Zoomed In)")
# plt.xlabel("n")
# plt.ylabel("value of peak in km/s")
# plt.xlim(-1, 10.0)
# plt.legend()
# plt.savefig("(zoomed) distribution 4-esc vel.png")
# plt.show()


