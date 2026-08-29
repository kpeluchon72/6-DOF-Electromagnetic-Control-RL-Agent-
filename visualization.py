"""VPython viewer. This module reads physics state but never advances it."""

import numpy as np
from vpython import arrow, box, canvas, color, cylinder, label, sphere, vector

from constants import (
    ARROW_LENGTH,
    COIL_RADIUS,
    MAGNET_LENGTH,
    MAGNET_RADIUS,
    MAX_CURRENT,
)


class SimulationViewer:
    """Visualize a Simulation without owning or changing its physics."""

    def __init__(
        self,
        simulation,
        width=1000,
        height=700,
        max_current=MAX_CURRENT,
        target_position=(0.0, 0.0, 0.0),
        target_orientation=None,
    ):
        self.simulation = simulation
        self.max_current = float(max_current)
        self.initial_camera_pos = vector(1, 0, 1)
        self.initial_camera_axis = vector(-1, 0, -1)
        self.scene = canvas(
            title="Electromagnet Sim",
            width=width,
            height=height,
            background=color.black,
        )
        self.reset_camera()
        self.scene.caption = (
            "S: pause/run | R: reset camera | 1-6: select coil | "
            "Up/Down: change current | 0: turn selected coil off\n"
            "Positive current is red, negative is blue, and zero is gray."
        )

        box(
            # The bottom coil spans y=-0.185 to -0.135, so its underside
            # remains visibly above the floor.
            pos=vector(0, -0.22, 0),
            size=vector(1, 0.05, 1),
            color=vector(150 / 255, 150 / 255, 150 / 255),
        )
        self.target_visual = sphere(
            pos=vector(*target_position), radius=0.005, color=color.red
        )
        self.target_orientation_visual = None
        self.set_target(target_position, target_orientation)

        magnetic_object = simulation.object
        direction = vector(*magnetic_object.magnetic_moment_direction)
        center = vector(*magnetic_object.position)
        self.moment_arrow = arrow(
            pos=center,
            axis=direction * ARROW_LENGTH,
            shaftwidth=0.003,
            color=color.yellow,
        )
        self.magnetic_object_visual = cylinder(
            pos=center - direction * (MAGNET_LENGTH / 2),
            axis=direction * MAGNET_LENGTH,
            radius=MAGNET_RADIUS,
            color=color.blue,
        )

        self.coil_visuals = []
        self.current_labels = []
        coil_length = 0.05
        for index, electromagnet in enumerate(simulation.electromagnets, start=1):
            coil_center = vector(*electromagnet.position)
            coil_direction = vector(*electromagnet.direction)
            self.coil_visuals.append(
                cylinder(
                    pos=coil_center - coil_direction * (coil_length / 2),
                    axis=coil_direction * coil_length,
                    radius=COIL_RADIUS,
                    color=self.current_color(electromagnet.current),
                )
            )
            self.current_labels.append(
                label(
                    pos=coil_center,
                    text=f"{index}: {electromagnet.current:+.0f} A",
                    height=10,
                    box=False,
                    opacity=0,
                    color=color.white,
                    yoffset=45,
                )
            )

    def reset_camera(self):
        self.scene.camera.pos = self.initial_camera_pos
        self.scene.camera.axis = self.initial_camera_axis
        self.scene.center = vector(0, 0, 0)

    def set_target(self, target_position, target_orientation=None):
        """Move the goal marker and optionally show its desired orientation."""
        target_center = vector(*target_position)
        self.target_visual.pos = target_center

        if target_orientation is None:
            if self.target_orientation_visual is not None:
                self.target_orientation_visual.visible = False
            return

        direction = np.asarray(target_orientation, dtype=float)
        direction /= np.linalg.norm(direction) + 1e-8
        target_axis = vector(*direction) * ARROW_LENGTH
        if self.target_orientation_visual is None:
            self.target_orientation_visual = arrow(
                pos=target_center,
                axis=target_axis,
                shaftwidth=0.003,
                color=color.green,
            )
        else:
            self.target_orientation_visual.visible = True
            self.target_orientation_visual.pos = target_center
            self.target_orientation_visual.axis = target_axis

    def current_color(self, current):
        """Return gray/off, red/positive, or blue/negative with intensity."""
        intensity = min(abs(float(current)) / self.max_current, 1.0)
        if np.isclose(current, 0.0):
            return vector(0.25, 0.25, 0.25)
        dim = 0.08 + 0.17 * intensity
        bright = 0.35 + 0.65 * intensity
        if current > 0:
            return vector(bright, dim, dim)
        return vector(dim, dim, bright)

    def update(self, selected_coil=None):
        magnetic_object = self.simulation.object
        direction = vector(*magnetic_object.magnetic_moment_direction)
        center = vector(*magnetic_object.position)
        self.magnetic_object_visual.pos = center - direction * (MAGNET_LENGTH / 2)
        self.magnetic_object_visual.axis = direction * MAGNET_LENGTH
        self.moment_arrow.pos = center
        self.moment_arrow.axis = direction * ARROW_LENGTH

        for index, (physics_coil, visual_coil, current_label) in enumerate(
            zip(
                self.simulation.electromagnets,
                self.coil_visuals,
                self.current_labels,
            )
        ):
            visual_coil.color = self.current_color(physics_coil.current)
            visual_coil.opacity = 1.0 if index == selected_coil else 0.75
            current_label.text = f"{index + 1}: {physics_coil.current:+.0f} A"
