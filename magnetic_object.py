import numpy as np


class MagneticObject:
    """
    Permanent magnetic object treated as a magnetic dipole.
    """

    def __init__(
        self,
        position,
        velocity,
        magnetic_moment_mag,
        mass,
        radius,
        length,
        magnetic_moment_direction=(0.01, -1.0, 0.0),
    ):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.angular_velocity = np.zeros(3)

        self.magnetic_moment_magnitude = magnetic_moment_mag
        direction = np.asarray(magnetic_moment_direction, dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            raise ValueError("Magnetic moment direction cannot be zero.")
        self.magnetic_moment_direction = direction / direction_norm

        self.mass = mass
        self.radius = radius
        self.length = length

        self.moment_of_inertia_parallel = 0.5 * self.mass * self.radius**2

        self.moment_of_inertia_perpendicular = (
            (1 / 12) * self.mass *
            (3 * self.radius**2 + self.length**2)
        )


    def gravitational_force(self, g=9.81):
        return np.array([0.0, -self.mass * g, 0.0])

    @property
    def magnetic_moment(self):
        return self.magnetic_moment_magnitude * self.magnetic_moment_direction
        
