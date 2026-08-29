"""Interactive VPython demo; physics and rendering live in separate modules."""

from vpython import rate

from constants import DT, MAX_CURRENT, SCREEN_HEIGHT, SCREEN_WIDTH, SUBSTEPS
from visualization import SimulationViewer
from world import create_default_world


CURRENT_STEP = 5.0


def run():
    # Start with the -y electromagnet at the configured current limit.
    simulation = create_default_world(currents=[0, 0, 0, MAX_CURRENT, 0, 0])
    viewer = SimulationViewer(
        simulation, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, max_current=MAX_CURRENT
    )
    state = {"running": False, "selected_coil": 0}

    def keydown(event):
        key = event.key.lower()
        if key == "r":
            viewer.reset_camera()
        elif key == "s":
            state["running"] = not state["running"]
        elif key in ("1", "2", "3", "4", "5", "6"):
            state["selected_coil"] = int(key) - 1
        elif key in ("up", "down", "0"):
            coil = simulation.electromagnets[state["selected_coil"]]
            if key == "0":
                coil.current = 0.0
            else:
                delta = CURRENT_STEP if key == "up" else -CURRENT_STEP
                coil.current = max(-MAX_CURRENT, min(MAX_CURRENT, coil.current + delta))
            viewer.update(state["selected_coil"])

    viewer.scene.bind("keydown", keydown)
    viewer.update(state["selected_coil"])

    while True:
        rate(60)
        if state["running"]:
            for _ in range(SUBSTEPS):
                simulation.step(DT)
            viewer.update(state["selected_coil"])


if __name__ == "__main__":
    run()
