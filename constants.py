import numpy as np


# Physics defaults
MU_0 = 4 * np.pi * 1e-7
DT = 0.001
MAGNET_RADIUS = 0.02
MAGNET_LENGTH = 0.005
MAGNET_MASS = 0.047
MAGNETIC_MOMENT = 6.0
SUBSTEPS = 17
BOX_HALF_SIZE = 0.13
DAMPING_COEFFICIENT = 0.05
ROTATIONAL_DAMPING_COEFFICIENT = 0.001

# Six large air-core coils surrounding the target volume.
COIL_DISTANCE = 0.16
COIL_RADIUS = 0.10
COIL_AREA = np.pi * COIL_RADIUS**2
COIL_TURNS = 1000

# Environment defaults
MAX_CURRENT = 20.0
MAX_CURRENT_CHANGE = 2.0
EPISODE_SECONDS = 10.0

# Viewer defaults. Physics code does not import or depend on these values.
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
ARROW_LENGTH = 0.03
