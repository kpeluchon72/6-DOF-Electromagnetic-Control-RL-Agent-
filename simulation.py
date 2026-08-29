import numpy as np
from constants import (
    BOX_HALF_SIZE,
    DAMPING_COEFFICIENT,
    MAGNET_LENGTH,
    MU_0,
    ROTATIONAL_DAMPING_COEFFICIENT,
)
from physics_kernel import NUMBA_AVAILABLE, advance_physics


class Simulation:

    def __init__(self, magnetic_object, electromagnets, box_half_size=BOX_HALF_SIZE):
        self.object = magnetic_object
        self.electromagnets = list(electromagnets)
        if self.electromagnets:
            self._coil_positions = np.stack(
                [coil.position for coil in self.electromagnets]
            )
            self._coil_moment_per_amp = np.stack(
                [coil.turns * coil.area * coil.direction for coil in self.electromagnets]
            )
        else:
            self._coil_positions = np.empty((0, 3), dtype=float)
            self._coil_moment_per_amp = np.empty((0, 3), dtype=float)

        object_extent = getattr(magnetic_object, "length", MAGNET_LENGTH)
        limit = float(box_half_size) - object_extent
        if limit <= 0:
            raise ValueError("The simulation box must be larger than the object.")
        self.box_min = np.full(3, -limit)
        self.box_max = np.full(3, limit)

    def magnetic_force(self):
        return self._magnetic_force_and_torque()[0]

    def _magnetic_force_and_torque(self):
        """Calculate all six coil interactions in one NumPy operation."""
        if not self.electromagnets:
            return np.zeros(3), np.zeros(3)

        displacement = self.object.position - self._coil_positions
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(distance < 1e-6):
            raise ValueError("Magnetic objects are too close.")

        direction = displacement / distance[:, None]
        currents = np.fromiter(
            (coil.current for coil in self.electromagnets),
            dtype=float,
            count=len(self.electromagnets),
        )
        coil_moments = self._coil_moment_per_amp * currents[:, None]
        object_moment = self.object.magnetic_moment

        coil_dot_direction = np.sum(coil_moments * direction, axis=1)
        object_dot_direction = direction @ object_moment
        coil_dot_object = coil_moments @ object_moment
        force_coefficient = 3.0 * MU_0 / (4.0 * np.pi * distance**4)
        forces = force_coefficient[:, None] * (
            coil_dot_direction[:, None] * object_moment
            + object_dot_direction[:, None] * coil_moments
            + coil_dot_object[:, None] * direction
            - 5.0
            * (coil_dot_direction * object_dot_direction)[:, None]
            * direction
        )

        field_coefficient = MU_0 / (4.0 * np.pi * distance**3)
        fields = field_coefficient[:, None] * (
            3.0 * coil_dot_direction[:, None] * direction - coil_moments
        )
        torque = np.cross(object_moment, np.sum(fields, axis=0))
        return np.sum(forces, axis=0), torque

    def total_force(self):
        damping_force = -DAMPING_COEFFICIENT * self.object.velocity

        return self.magnetic_force() + self.object.gravitational_force() + damping_force

    def total_torque(self):
        total_torque = self._magnetic_force_and_torque()[1]
        rotational_damping = (
            -ROTATIONAL_DAMPING_COEFFICIENT * self.object.angular_velocity
        )

        return total_torque + rotational_damping

    @staticmethod
    def rotate_vector(v, axis, angle):
        axis = axis / np.linalg.norm(axis)

        return (
            v * np.cos(angle)
            + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1 - np.cos(angle))
        )


        

    def step(self, dt):
        """
        Advance simulation by dt seconds.

        Uses semi-implicit Euler integration.
        """

        if dt <= 0:
            raise ValueError("dt must be positive.")

        magnetic_force, magnetic_torque = self._magnetic_force_and_torque()
        damping_force = -DAMPING_COEFFICIENT * self.object.velocity
        force = magnetic_force + self.object.gravitational_force() + damping_force

        # Newton's second law
        acceleration = force / self.object.mass

        # Update velocity first
        self.object.velocity += acceleration * dt

        # Then position
        self.object.position += self.object.velocity * dt




        rotational_damping = (
            -ROTATIONAL_DAMPING_COEFFICIENT * self.object.angular_velocity
        )
        torque = magnetic_torque + rotational_damping

        n = self.object.magnetic_moment_direction

        tau_parallel = np.dot(torque, n) * n
        tau_perpendicular = torque - tau_parallel

        angular_acceleration = (
            tau_parallel / self.object.moment_of_inertia_parallel
            + tau_perpendicular / self.object.moment_of_inertia_perpendicular
        ) 

        self.object.angular_velocity += angular_acceleration * dt
        omega = self.object.angular_velocity
        omega_mag = np.linalg.norm(omega)

        if omega_mag > 0:
            angle = omega_mag * dt
            axis = omega / omega_mag

            new_direction = self.rotate_vector(
                self.object.magnetic_moment_direction,
                axis,
                angle
            )

            self.object.magnetic_moment_direction = (
                new_direction / np.linalg.norm(new_direction)
            )


        # Keep object inside box
        for i in range(3):

            # Hit negative wall
            if self.object.position[i] < self.box_min[i]:
                self.object.position[i] = self.box_min[i]

                if self.object.velocity[i] < 0:
                    self.object.velocity[i] = 0

            # Hit positive wall
            elif self.object.position[i] > self.box_max[i]:
                self.object.position[i] = self.box_max[i]

                if self.object.velocity[i] > 0:
                    self.object.velocity[i] = 0

    def step_many(self, dt, substeps):
        """Advance multiple steps through the compiled RL training path."""
        if dt <= 0 or substeps <= 0:
            raise ValueError("dt and substeps must be positive.")
        if not NUMBA_AVAILABLE:
            for _ in range(substeps):
                self.step(dt)
            return

        currents = np.fromiter(
            (coil.current for coil in self.electromagnets),
            dtype=float,
            count=len(self.electromagnets),
        )
        advance_physics(
            self.object.position,
            self.object.velocity,
            self.object.magnetic_moment_direction,
            self.object.angular_velocity,
            self.object.mass,
            self.object.magnetic_moment_magnitude,
            self.object.moment_of_inertia_parallel,
            self.object.moment_of_inertia_perpendicular,
            self._coil_positions,
            self._coil_moment_per_amp,
            currents,
            self.box_min,
            self.box_max,
            float(dt),
            int(substeps),
            DAMPING_COEFFICIENT,
            ROTATIONAL_DAMPING_COEFFICIENT,
            MU_0,
        )
