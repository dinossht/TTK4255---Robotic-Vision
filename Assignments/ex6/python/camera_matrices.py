import numpy as np

def camera_matrices(K1, K2, R, t):
    """ Computes the projection matrix for camera 1 and camera 2.

    Args:
        K1,K2: Intrinsic matrix for camera 1 and camera 2.
        R,t: The rotation and translation mapping points in camera 1 to points in camera 2.

    Returns:
        P1,P2: The projection matrices with shape 3x4.
    """
    # P = K [I 0] and P' = K' [R t]
    P1 = K1 @ np.block([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.block([R, t.reshape(3, 1)])
    return P1, P2
