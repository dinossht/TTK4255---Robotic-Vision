import matplotlib.pyplot as plt
import numpy as np
from common import draw_frame


# Every equations that are referenced in this file is based on:
# "Multiple View Geometry in Computer Vision - 2nd edition"


def estimate_H(xy, XY):
    #
    # Task 2: Implement estimate_H
    #
    A = np.zeros((xy.shape[0]*xy.shape[1], 9))
    # Based on Algorithm 4.1
    for r in range(xy.shape[0]):
        # Equation 14 from assignment
        x = xy[r, 0]
        y = xy[r, 1]
        X = XY[r, 0]
        Y = XY[r, 1]
        A[r*2, :] = np.array([X, Y, 1, 0, 0, 0, -X*x, -Y*x, -x])
        A[r*2+1, :] = np.array([0, 0, 0, X, Y, 1, -X*y, -Y*y, -y])

    # Obtain the SVD of A (section A4.4(p585)). The unit singular vector
    # corresponding to the smallest singular value is the solution h.
    # Specifically, if A = UDV T with D diagonal with positive diagonal
    # entries, arranged in descending order down the diagonal, then h
    # is the last column of V.
    _, _, V = np.linalg.svd(A)
    # Equation 4.2 from textbook
    H = V[-1, :].reshape(3, 3)
    return H


def decompose_H(H):
    #
    # Task 3a: Implement decompose_H
    #
    # Initialize 
    T1 = np.eye(4)
    T2 = np.eye(4)
    # Equation (17)-(18) from assignment
    # The third column of the rotation matrix is not present,
    # but if we know two columns, the third can always be
    # obtained using the cross product
    lamda = np.linalg.norm(H[:, 0])

    # Extract rotation
    r1 = H[:, 0] / lamda
    r2 = H[:, 1] / lamda
    r3 = np.cross(r1, r2)
    # Extract translation vector
    t = H[:, 2] / lamda
    T1[0:3, 0:4] = np.column_stack((r1, r2, r3, t))

    # Extract rotation
    r1 = H[:, 0] / -lamda
    r2 = H[:, 1] / -lamda
    r3 = np.cross(r1, r2)
    # Extract translation vector
    t = H[:, 2] / -lamda
    T2[0:3, 0:4] = np.column_stack((r1, r2, r3, t))

    return T1, T2


def choose_solution(T1, T2):
    #
    # Task 3b: Implement choose_solution
    #
    tz = T1[2, 3]
    # tz positive means the object that
    # is seen by the camera is in front of it
    return T1 if (tz >= 0) else T2


K = np.loadtxt('../data/cameraK.txt')
all_markers = np.loadtxt('../data/markers.txt')
XY = np.loadtxt('../data/model.txt')
n = len(XY)

for image_number in range(23):
    I = plt.imread('../data/video%04d.jpg' % image_number)
    markers = all_markers[image_number, :]
    markers = np.reshape(markers, [n, 3])
    # First column is 1 if marker was detected
    matched = markers[:, 0].astype(bool)
    uv = markers[matched, 1:3]  # Get markers for which matched = 1

    # Convert pixel coordinates to normalized image coordinates
    xy = (uv - K[0:2, 2])/np.array([K[0, 0], K[1, 1]])

    H = estimate_H(xy, XY[matched, :2])
    T1, T2 = decompose_H(H)
    T = choose_solution(T1, T2)

    # Compute predicted corner locations using model and homography
    uv_hat = (K@H@XY.T)
    uv_hat = (uv_hat/uv_hat[2, :]).T

    plt.clf()
    plt.imshow(I, interpolation='bilinear')
    draw_frame(K, T, scale=7)
    plt.scatter(uv[:, 0], uv[:, 1], color='red', label='Observed')
    plt.scatter(uv_hat[:, 0], uv_hat[:, 1], marker='+',
                color='yellow', label='Predicted')
    plt.legend()
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])
    plt.savefig('../data/out%04d.png' % image_number)
