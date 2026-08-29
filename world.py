"""Factories for the physical world, with no visualization dependencies."""

from constants import (
    MAGNETIC_MOMENT,
    MAGNET_LENGTH,
    MAGNET_MASS,
    MAGNET_RADIUS,
)
from electromagnet import create_magnets
from magnetic_object import MagneticObject
from simulation import Simulation


def create_default_world(currents=None):
    """Return a fresh simulation in the default initial state."""
    electromagnets = create_magnets(currents=currents)
    magnetic_object = MagneticObject(
        position=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        magnetic_moment_mag=MAGNETIC_MOMENT,
        radius=MAGNET_RADIUS,
        length=MAGNET_LENGTH,
        mass=MAGNET_MASS,
    )
    return Simulation(magnetic_object, electromagnets)
