"""Compiled inner loop for headless RL physics stepping."""

import math

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(function):
            return function

        return decorator


@njit(cache=True)
def advance_physics(
    position,
    velocity,
    moment_direction,
    angular_velocity,
    mass,
    moment_magnitude,
    inertia_parallel,
    inertia_perpendicular,
    coil_positions,
    coil_moment_per_amp,
    currents,
    box_min,
    box_max,
    dt,
    substeps,
    damping,
    rotational_damping,
    mu_0,
):
    """Advance several semi-implicit Euler steps without Python allocations."""
    magnetic_constant = mu_0 / (4.0 * math.pi)

    for _ in range(substeps):
        m2x = moment_magnitude * moment_direction[0]
        m2y = moment_magnitude * moment_direction[1]
        m2z = moment_magnitude * moment_direction[2]
        force_x = 0.0
        force_y = 0.0
        force_z = 0.0
        field_x = 0.0
        field_y = 0.0
        field_z = 0.0

        for index in range(coil_positions.shape[0]):
            rx = position[0] - coil_positions[index, 0]
            ry = position[1] - coil_positions[index, 1]
            rz = position[2] - coil_positions[index, 2]
            distance_squared = rx * rx + ry * ry + rz * rz
            distance = math.sqrt(distance_squared)
            if distance < 1e-6:
                raise ValueError("Magnetic objects are too close.")

            inv_distance = 1.0 / distance
            nx = rx * inv_distance
            ny = ry * inv_distance
            nz = rz * inv_distance
            current = currents[index]
            m1x = coil_moment_per_amp[index, 0] * current
            m1y = coil_moment_per_amp[index, 1] * current
            m1z = coil_moment_per_amp[index, 2] * current

            m1_dot_n = m1x * nx + m1y * ny + m1z * nz
            m2_dot_n = m2x * nx + m2y * ny + m2z * nz
            m1_dot_m2 = m1x * m2x + m1y * m2y + m1z * m2z
            force_factor = 3.0 * magnetic_constant / (distance_squared**2)
            common = m1_dot_m2 - 5.0 * m1_dot_n * m2_dot_n
            force_x += force_factor * (
                m1_dot_n * m2x + m2_dot_n * m1x + common * nx
            )
            force_y += force_factor * (
                m1_dot_n * m2y + m2_dot_n * m1y + common * ny
            )
            force_z += force_factor * (
                m1_dot_n * m2z + m2_dot_n * m1z + common * nz
            )

            field_factor = magnetic_constant / (distance_squared * distance)
            field_x += field_factor * (3.0 * m1_dot_n * nx - m1x)
            field_y += field_factor * (3.0 * m1_dot_n * ny - m1y)
            field_z += field_factor * (3.0 * m1_dot_n * nz - m1z)

        force_x -= damping * velocity[0]
        force_y -= mass * 9.81 + damping * velocity[1]
        force_z -= damping * velocity[2]
        velocity[0] += force_x / mass * dt
        velocity[1] += force_y / mass * dt
        velocity[2] += force_z / mass * dt
        position[0] += velocity[0] * dt
        position[1] += velocity[1] * dt
        position[2] += velocity[2] * dt

        torque_x = m2y * field_z - m2z * field_y - rotational_damping * angular_velocity[0]
        torque_y = m2z * field_x - m2x * field_z - rotational_damping * angular_velocity[1]
        torque_z = m2x * field_y - m2y * field_x - rotational_damping * angular_velocity[2]
        parallel_scale = (
            torque_x * moment_direction[0]
            + torque_y * moment_direction[1]
            + torque_z * moment_direction[2]
        )
        parallel_x = parallel_scale * moment_direction[0]
        parallel_y = parallel_scale * moment_direction[1]
        parallel_z = parallel_scale * moment_direction[2]
        angular_velocity[0] += (
            parallel_x / inertia_parallel
            + (torque_x - parallel_x) / inertia_perpendicular
        ) * dt
        angular_velocity[1] += (
            parallel_y / inertia_parallel
            + (torque_y - parallel_y) / inertia_perpendicular
        ) * dt
        angular_velocity[2] += (
            parallel_z / inertia_parallel
            + (torque_z - parallel_z) / inertia_perpendicular
        ) * dt

        omega_squared = (
            angular_velocity[0] ** 2
            + angular_velocity[1] ** 2
            + angular_velocity[2] ** 2
        )
        if omega_squared > 0.0:
            omega = math.sqrt(omega_squared)
            ax = angular_velocity[0] / omega
            ay = angular_velocity[1] / omega
            az = angular_velocity[2] / omega
            angle = omega * dt
            cosine = math.cos(angle)
            sine = math.sin(angle)
            dot = (
                ax * moment_direction[0]
                + ay * moment_direction[1]
                + az * moment_direction[2]
            )
            cross_x = ay * moment_direction[2] - az * moment_direction[1]
            cross_y = az * moment_direction[0] - ax * moment_direction[2]
            cross_z = ax * moment_direction[1] - ay * moment_direction[0]
            one_minus_cosine = 1.0 - cosine
            new_x = (
                moment_direction[0] * cosine
                + cross_x * sine
                + ax * dot * one_minus_cosine
            )
            new_y = (
                moment_direction[1] * cosine
                + cross_y * sine
                + ay * dot * one_minus_cosine
            )
            new_z = (
                moment_direction[2] * cosine
                + cross_z * sine
                + az * dot * one_minus_cosine
            )
            direction_length = math.sqrt(new_x * new_x + new_y * new_y + new_z * new_z)
            moment_direction[0] = new_x / direction_length
            moment_direction[1] = new_y / direction_length
            moment_direction[2] = new_z / direction_length

        for axis in range(3):
            if position[axis] < box_min[axis]:
                position[axis] = box_min[axis]
                if velocity[axis] < 0.0:
                    velocity[axis] = 0.0
            elif position[axis] > box_max[axis]:
                position[axis] = box_max[axis]
                if velocity[axis] > 0.0:
                    velocity[axis] = 0.0
