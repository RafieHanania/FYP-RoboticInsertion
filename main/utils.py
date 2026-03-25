import math
import numpy as np

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.y = None

    def update(self, x):
        if self.y is None:
            self.y = x
        else:
            self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y

def rotvec_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Convert a UR-style rotation vector (axis-angle / Rodrigues) to a 3x3
    rotation matrix.
 
    The UR `actual_tcp_pose` reports orientation as [rx, ry, rz] where the
    direction of the vector is the rotation axis and its magnitude is the
    angle in radians.
 
    Returns R_base_to_tcp such that:
        p_base = R_base_to_tcp @ p_tcp   (for points)
        v_base = R_base_to_tcp @ v_tcp   (for free vectors / velocities)
    """
    rotvec = np.array([rx, ry, rz], dtype=float)
    angle = np.linalg.norm(rotvec)
 
    if angle < 1e-12:
        return np.eye(3)
 
    # Unit axis
    k = rotvec / angle
 
    # Skew-symmetric matrix of k
    K = np.array([
        [ 0.0, -k[2],  k[1]],
        [ k[2], 0.0,  -k[0]],
        [-k[1], k[0],  0.0 ],
    ])
 
    # Rodrigues' formula:  R = I + sin(θ) K + (1 - cos(θ)) K²
    R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return R
 
 
def vel_tcp_to_base(v_tcp_6: np.ndarray,
                    rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Rotate a 6-DOF velocity from TCP frame to base frame.
 
    Parameters
    ----------
    v_tcp_6 : (6,) array  [vx, vy, vz, wx, wy, wz] in TCP frame
    rx, ry, rz : float    rotation-vector components from actual_tcp_pose
 
    Returns
    -------
    v_base_6 : (6,) array [vx, vy, vz, wx, wy, wz] in base frame
 
    Notes
    -----
    For a rigid body velocity expressed at the same point (the TCP origin),
    both the linear and angular parts transform by the *same* rotation
    matrix R_base_tcp.  This is correct because speedl() expects the
    velocity of the TCP origin expressed in the base frame, which is
    exactly R @ v_linear_tcp, and angular velocity transforms the same way.
    """
    R = rotvec_to_matrix(rx, ry, rz)           # R_base_to_tcp  (3x3)
 
    v_base_6 = np.empty(6)
    v_base_6[:3] = R @ v_tcp_6[:3]             # linear velocity
    v_base_6[3:] = R @ v_tcp_6[3:]             # angular velocity
    return v_base_6
 