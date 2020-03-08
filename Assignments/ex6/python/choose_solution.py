import numpy as np
from linear_triangulation import *
from camera_matrices import *

def choose_solution(uv1, uv2, K1, K2, Rts):
    """
    Chooses among the rotation and translation solutions Rts
    the one which gives the most points in front of both cameras.
    """

    positive_Z_count = np.zeros((4, 1))
    # Iterate over all points and all possible R,t combinations and 
    # count which R,t combination gives most positive Z-points
    for i in range(len(Rts)):
        for j in range(len(uv1)):
            R, t = Rts[i]
            P1, P2 = camera_matrices(K1, K2, R, t) 
            X = linear_triangulation(uv1[j], uv2[j], P1, P2)
            
            # Count how many points have positive Z coordinate
            if X[-1] > 0:
                positive_Z_count[i] += 1      

    soln = positive_Z_count.argmax()
    print('Choosing solution %d' % soln)
    return Rts[soln]
