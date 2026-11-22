import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

def build_grid() -> tuple[Array, Array]:
    AR = np.linspace(4, 10, 30)[:, None]
    V = np.linspace(5, 25, 30)[None, :]

    #AR = np.array([[5.0]])
    #V = np.array([[15.08242389]])

    return AR, V