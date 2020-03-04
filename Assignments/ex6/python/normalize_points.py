import numpy as np

def normalize_points(pts):
    """ Computes a normalizing transformation of the points such that
    the points are centered at the origin and their mean distance from
    the origin is equal to sqrt(2).

    See HZ, Ch. 4.4.4: Normalizing transformations (p107).

    Args:
        pts:    Input 2D point array of shape n x 2

    Returns:
        pts_n:  Normalized 2D point array of shape n x 2
        T:      The normalizing transformation in 3x3 matrix form, such
                that for a point (x,y), the normalized point (x',y') is
                found by multiplying T with the point:

                    |x'|       |x|
                    |y'| = T * |y|
                    |1 |       |1|
    """

    # todo: Compute pts_n and T
    mu = np.mean(pts, axis=0)
    sigma = np.mean(np.linalg.norm(pts - mu, axis=1))
    pts_n = pts

    s = np.sqrt(2) / sigma
    T = np.array([
        [s, 0, -s*mu[0]],
        [0, s, -s*mu[1]],
        [0, 0, 1]
        ])
    pts_temp = T@np.column_stack((pts, np.ones(len(pts)))).T
    pts_n = np.divide(pts_temp[0:2], pts_temp[2]).T
    return pts_n, T
