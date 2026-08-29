import numpy as np
from constants import (
    COIL_AREA,
    COIL_DISTANCE,
    COIL_TURNS,
    MAX_CURRENT,
    MU_0,
)

class Electromagnet:
    """
    Electromagnet approximated as a magnetic dipole.

    position : center of electromagnet [m]
    direction: direction its north pole points
    turns    : number of coil turns
    area     : coil cross-sectional area [m^2]
    current  : current through coil [A]
    """

    def __init__(self, position, direction, turns, area, current=0.0):
        self.position = np.array(position, dtype=float)

        direction = np.array(direction, dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            raise ValueError("Electromagnet direction cannot be zero.")
        self.direction = direction / direction_norm

        self.turns = turns
        self.area = area
        self.current = current

    @property
    def magnetic_moment(self):
        """
        Magnetic dipole moment:
            m = N I A n

        Units: A m^2
        """
        return (
            self.turns
            * self.current
            * self.area
            * self.direction
        )

    def magnetic_field(self, point):
        """
        Magnetic field produced by the electromagnet
        at a point in space.

        Dipole approximation:
            B = mu0/(4pi r^3)
                [3(m·r_hat)r_hat - m]
        """

        r = np.array(point, dtype=float) - self.position
        distance = np.linalg.norm(r)

        if distance < 1e-6:
            raise ValueError("Point is too close to electromagnet.")

        r_hat = r / distance
        m = self.magnetic_moment

        return (
            MU_0 / (4 * np.pi * distance**3)
            * (
                3 * np.dot(m, r_hat) * r_hat
                - m
            )
        )

    def force_on(self, magnetic_object):
        """
        Force exerted by this electromagnet
        on another magnetic dipole.

        This is the exact dipole-dipole force
        under the point-dipole approximation.
        """

        r = magnetic_object.position - self.position
        distance = np.linalg.norm(r)

        if distance < 1e-6:
            raise ValueError("Magnetic objects are too close.")

        r_hat = r / distance

        m1 = self.magnetic_moment
        m2 = magnetic_object.magnetic_moment

        m1_dot_r = np.dot(m1, r_hat)
        m2_dot_r = np.dot(m2, r_hat)
        m1_dot_m2 = np.dot(m1, m2)

        coefficient = (
            3 * MU_0
            / (4 * np.pi * distance**4)
        )

        return coefficient * (
            m1_dot_r * m2
            + m2_dot_r * m1
            + m1_dot_m2 * r_hat
            - 5 * m1_dot_r * m2_dot_r * r_hat
        )


    def torque_on(self, magnetic_object):
        """
        Torque on the magnetic object:

            tau = m x B
        """

        B = self.magnetic_field(magnetic_object.position)

        return np.cross(
            magnetic_object.magnetic_moment,
            B
        )


def create_magnets(distance=COIL_DISTANCE, currents=None):
    """Create six coils in +x, -x, +y, -y, +z, -z order."""
    if currents is None:
        # Preserve the original interactive demo's powered -y coil.
        currents = np.array([0, 0, 0, MAX_CURRENT, 0, 0], dtype=float)
    currents = np.asarray(currents, dtype=float)
    if currents.shape != (6,):
        raise ValueError("currents must contain exactly six values.")

    positions_and_directions = (
        ([distance, 0, 0], [-1, 0, 0]),
        ([-distance, 0, 0], [1, 0, 0]),
        ([0, distance, 0], [0, -1, 0]),
        ([0, -distance, 0], [0, 1, 0]),
        ([0, 0, distance], [0, 0, -1]),
        ([0, 0, -distance], [0, 0, 1]),
    )

    return [
        Electromagnet(
            position,
            direction,
            turns=COIL_TURNS,
            area=COIL_AREA,
            current=current,
        )
        for (position, direction), current in zip(positions_and_directions, currents)
    ]
