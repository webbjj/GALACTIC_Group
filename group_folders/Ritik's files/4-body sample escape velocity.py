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


# def _sample_escape_velocity_four_body(e0, ms, mb, n, vs, npeak=5, nrandom=1000):  # add new parameters (might need vs)

def _sample_escape_velocity_four_body(e0, ms, mb, n, npeak=5, nrandom=1000):
    # randomly sample between npeak * vs_peak

    vs_peak = _escape_velocity_distribution_peak_four_body(n, e0, ms, mb)
    match = False

    while not match:
        vstemp = np.random.rand(nrandom) * npeak * vs_peak
        amptemp = np.random.rand(nrandom) * vs_peak

        aindx = amptemp < _escape_velocity_distribution_four_body(
                    n, np.ones(nrandom) * vstemp, np.ones(nrandom) * e0,
                    np.ones(nrandom) * ms, np.ones(nrandom) * mb)

        if np.sum(aindx) > 0:
            vs = vstemp[aindx][0]
            match = True

    return vs


def _escape_velocity_distribution_four_body(n, vs, e0, ms, mb):
    # Equation 24 from Leigh et al., 2021
    M = ms + mb
    fv = ((n - 1) * (np.fabs(e0) ** (n - 1)) * ms * M / mb) * vs / \
         ((np.fabs(e0) + 0.5 * (ms * M / mb) * (vs ** 2.)) ** n)
    return fv


def _escape_velocity_distribution_peak_four_body(n, e0, ms, mb):
    # Equation 25 from Leigh et al., 2021
    M = ms + mb
    epsilon = (n - 0.5) ** (-0.5)
    vs_peak = epsilon * ((M - ms) / (ms * M)) ** 0.5 * (np.fabs(e0)) ** 0.5

    return vs_peak


m_single = 4.06  # Mass [solar masses]
m1 = 3.98
m2 = 4.61
m3 = 4.75
m_triplet = m1 + m2 + m3
m_total = m_triplet + m_single
energy_single = 1.5e10  # kinetic energy
energy_triplet = 5e10
energy_total = energy_triplet + energy_single


# m_0 is average mass of four bodies
def m_naught(m_a, m_b, m_c, m_s):  # modified eqn 7.28 from Valtonen 2003
    return ((m_a * (m_b + m_c) + m_b * m_c + m_s * (m_a + m_b + m_c)) / 4)**0.5


# L_0 is total angular momentum, normalized by L = L_0 / L_max
# L_max is maximum angular momentum of system
L_max = 2.5 * sp.constants.G * (m_naught(m1, m2, m3, m_single)**5 /
                                np.fabs(energy_total))**0.5

L1 = 0 / L_max
L2 = 0.4 / L_max
L3 = 0.15 / L_max

n1 = 18 * L1**2 + 3  # eqn for n from Leigh et al., 2021


# velocities = []
# i = 0
# while i < 10:
#     a = _sample_escape_velocity_four_body(energy_total, m_single, m_triplet,
#                                           n1)
#     velocities.append(a)
#     i += 1
# # vs_four = _sample_escape_velocity_four_body(energy_total, m_single, m_triplet,
# #                                             n1)
# print("four body", velocities)
#
#
# plt.hist(velocities, bins=7, color='#8d99ae', density=True)
print(n1)



