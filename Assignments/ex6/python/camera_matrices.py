import numpy as np

def camera_matrices(K1, K2, R, t):
    """ Computes the projection matrix for camera 1 and camera 2.

    Args:
        K1,K2: Intrinsic matrix for camera 1 and camera 2.
        R,t: The rotation and translation mapping points in camera 1 to points in camera 2.

    Returns:
        P1,P2: The projection matrices with shape 3x4.
    """

    # todo: compute the correct P1 and P2
    P1 = np.zeros((3,4))
    P2 = np.zeros((3,4))
    return P1, P2
