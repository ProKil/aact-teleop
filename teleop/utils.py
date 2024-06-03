import numpy as np

__all__ = ["_normalize_angle"]


def _normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-pi, pi].

    Args:
        angle: The angle to normalize.

    Returns:
        The normalized angle.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
