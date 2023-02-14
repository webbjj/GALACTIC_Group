from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, PlummerPotential, KingPotential, \
    MovingObjectPotential
from galpy.util import conversion, coords
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

import time

m_single = 4.06  # Mass [solar masses]
m_triplet = 12.5  # Mass [solar masses]
m_total = m_triplet + m_single
energy_single = 1.5e10  # kinetic energy
energy_triplet = 5e10
energy_total = energy_triplet + energy_single

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


# vs_three = _sample_escape_velocity_three_body(energy_total, m_single, m_triplet)
# print("three body", vs_three)
