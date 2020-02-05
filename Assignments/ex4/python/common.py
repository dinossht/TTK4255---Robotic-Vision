import numpy as np

K                  = np.loadtxt('../data/cameraK.txt')
p_model            = np.loadtxt('../data/model.txt')
platform_to_camera = np.loadtxt('../data/pose.txt')

def residuals(uv, weights, yaw, pitch, roll):
    # uv:       pixel coordinates of markers
    # weights:  weight = 1 if marker was detected or 0 otherwise   
    # rest:     quanser orientation

    # Helicopter model from Exercise 1 (you don't need to modify this).
    base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
    hinge_to_base    = translate(0, 0, 0.325)@rotate_y(pitch)
    arm_to_hinge     = translate(0, 0, -0.0552)
    rotors_to_arm    = translate(0.653, 0, -0.0312)@rotate_x(roll)
    base_to_camera   = platform_to_camera@base_to_platform
    hinge_to_camera  = base_to_camera@hinge_to_base
    arm_to_camera    = hinge_to_camera@arm_to_hinge
    rotors_to_camera = arm_to_camera@rotors_to_arm

    #
    # Task 1a: Implement the rest of this function
    #

    # Tip: If A is an Nx2 array, np.linalg.norm(A, axis=1)
    # computes the Euclidean length of each row of A and
    # returns an Nx1 array.

    m = p_model.shape[0] # Placeholder



    return np.ones(m)*np.inf # Placeholder

def normal_equations(uv, weights, yaw, pitch):
    #
    # Task 1b: Compute the normal equation terms
    #
    JTJ = np.eye(3)   # Placeholder
    JTr = np.zeros(3) # Placeholder
    return JTJ, JTr

def gauss_newton(uv, weights, yaw, pitch, roll):
    #
    # Task 1c: Implement the Gauss-Newton method
    #
    max_iter = 100
    step_size = 0.25
    for iter in range(max_iter):
        pass # Placeholder
    return yaw, pitch, roll

def levenberg_marquardt(uv, weights, yaw, pitch, roll):
    #
    # Task 2a: Implement the Levenberg-Marquardt method
    #
    return yaw, pitch, roll

def rotate_x(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0, 0],
                     [0, c,-s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])

def rotate_y(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rotate_z(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])
