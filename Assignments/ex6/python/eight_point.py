import numpy as np
from normalize_points import *

def eight_point(uv1, uv2):
    """ Given n >= 8 point matches, (u1 v1) <-> (u2 v2), compute the
    fundamental matrix F that satisfies the equations

        (u2 v2 1)^T * F * (u1 v1 1) = 0

    Args:
        uv1: (n x 2 array) Pixel coordinates in image 1.
        uv2: (n x 2 array) Pixel coordinates in image 2.

    Returns:
        F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1
             to lines in image 2.

    See HZ Ch. 11.2: The normalized 8-point algorithm (p.281).
    """

    # Normalization
    uv1, T1 = normalize_points(uv1)
    uv2, T2 = normalize_points(uv2)

    # Find fundemental matrix F_hat' 
    # (a) Linear solution:
    A = np.column_stack((
        uv2[:, 0] * uv1[:, 0], 
        uv2[:, 0] * uv1[:, 1], 
        uv2[:, 0], 
        uv2[:, 1] * uv1[:, 0],
        uv2[:, 1] * uv1[:, 1],
        uv2[:, 1],
        uv1[:, 0],
        uv1[:, 1],
        np.ones((len(uv1), 1)) 
    ))

    U, S, VT = np.linalg.svd(A)
    F_hat = np.reshape(VT.T[:, -1], (3, 3))

    """
    # (b) Constraint enforcement:
    U, S, VT = np.linalg.svd(F_hat)
    S = np.diag(S)
    S[-1, -1] = 0
    F_hat_marked = U@S@VT
    """
    
    # Denomalization
    F = T2.T@F_hat@T1
    return F

def closest_fundamental_matrix(F):
    """
    Computes the closest fundamental matrix in the sense of the
    Frobenius norm. See HZ, Ch. 11.1.1 (p.280).
    """

    # todo: Compute the correct F
    return F
