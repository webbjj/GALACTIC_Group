from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, PlummerPotential, KingPotential, \
    MovingObjectPotential
from galpy.util import conversion, coords
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

import time

m_single = 3  # Mass [solar masses]
m1 = 2
m2 = 1.5
m3 = 1.25
m_triplet = m1 + m2 + m3
m_total = m_triplet + m_single

relative_vel = 10  # km/s
semi = 10  # AU
energy_single = (1/2) * m_single * relative_vel**2 # kinetic energy FIX

energy_b_a = -(sp.constants.G * m_single * m1) / 2 * semi
energy_b_b = -(sp.constants.G * m2 * m3) / 2 * semi
energy_triplet = energy_b_a + energy_b_b
energy_total = energy_triplet + energy_single

m_single = 3  # Mass [solar masses]
m1 = 2
m2 = 1.5
m3 = 1.25
m_binary = m1 + m2
m_total = m_binary + m_single
energy_single = 1.5e10  # kinetic energy
energy_binary = 5e10
energy_total = energy_binary + energy_single

# vs_four = _sample_escape_velocity_four_body(energy_total, m_single, m_triplet, 4)
# print("four body", vs_four)


def _sample_escape_velocity_three_body(e0, ms, mb, npeak=5, nrandom=1000): # new version for 4 body
    # randomly sample between npeak*vs_peak

    vs_peak = _escape_velocity_distribution_peak(e0, ms, mb)
    match = False

    while not match:
        vstemp = np.random.rand(nrandom) * npeak * vs_peak
        amptemp = np.random.rand(nrandom) * vs_peak

        aindx = amptemp < _escape_velocity_distribution(
            np.ones(nrandom) * vstemp, np.ones(nrandom) * e0,
            np.ones(nrandom) * ms, np.ones(nrandom) * mb)

        if np.sum(aindx) > 0:
            vs = vstemp[aindx][0]
            match = True

    return vs


def _escape_velocity_distribution(vs, e0, ms, mb):
    # Equation 7.19
    M = ms + mb
    fv = (3.5 * (np.fabs(e0) ** (7. / 2.)) * ms * M / mb) * vs / (
            (np.fabs(e0) + 0.5 * (ms * M / mb) * (vs ** 2.)) ** (
            9. / 2.))
    return fv


def _escape_velocity_distribution_peak(e0, ms, mb):
    M = ms + mb
    vs_peak = 0.5 * np.sqrt((M - ms) / (ms * M)) * np.sqrt(np.fabs(e0))

    return vs_peak


velocities = []
i = 0
while i < 10:
    a = _sample_escape_velocity_three_body(energy_total, m_single, m_binary)
    velocities.append(a)
    i += 1

# vs_three = _sample_escape_velocity_three_body(energy_total, m_single, m_triplet,
#                                             n1)
# print("three body", velocities)

# print(_sample_escape_velocity_three_body(energy_total, m_single, m_binary))

plt.hist(velocities, bins=7, color='#8d99ae', density=True)
plt.xlabel("esc. velocity (km/s)")
plt.ylabel("normalized count")
plt.title("Sample Escape Velocities for 3-body Function")
plt.savefig("escvel(3body).png")
plt.show()


# vs_three = _sample_escape_velocity_three_body(energy_total, m_single, m_triplet)
# print("three body", vs_three)

