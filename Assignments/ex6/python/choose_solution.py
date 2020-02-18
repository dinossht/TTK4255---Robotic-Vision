import numpy as np
from linear_triangulation import *
from camera_matrices import *

def choose_solution(uv1, uv2, K1, K2, Rts):
    """
    Chooses among the rotation and translation solutions Rts
    the one which gives the most points in front of both cameras.
    """

    # todo: Choose the correct solution
    soln = 0
    print('Choosing solution %d' % soln)
    return Rts[soln]
