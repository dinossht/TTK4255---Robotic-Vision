import numpy as np

K                  = np.loadtxt('../data/cameraK.txt')
PI                 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
p_model            = np.loadtxt('../data/model.txt')
platform_to_camera = np.loadtxt('../data/pose.txt')

def residuals(uv, weights, yaw, pitch, roll):
    # uv:       pixel coordinates of markers
    # weights:  weight = 1 if marker was detected or 0 otherwise   
    # the rest:     quanser orientation

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
    
    r = np.zeros(7)
    v = np.zeros((7, 3))

    for i in range(len(uv)):
        if weights[i] == 1: 
            # markers: [0-2]  
            if i <= 2:
                v = K@PI@arm_to_camera@p_model[i]
            # markers: [3-6]        
            else:  
                v = K@PI@rotors_to_camera@p_model[i]
            
            r[i] = np.linalg.norm(v[0:2] / v[2] - uv[i, :])
    
    # Return residual vector
    return r

def normal_equations(uv, weights, yaw, pitch, roll):
    #
    # Task 1b: Compute the normal equation terms
    #
    J = np.zeros((7, 3))
    # Delta
    d = 0.01
    r = residuals(uv, weights, yaw, pitch, roll)
    
    # Gradient wrt. yaw
    r_y = residuals(uv, weights, yaw + d, pitch, roll)
    grad_y = (r_y - r) / d
    # Gradient wrt. pitch 
    r_p = residuals(uv, weights, yaw, pitch + d, roll)
    grad_p = (r_p - r) / d
    # Gradient wrt. roll 
    r_r = residuals(uv, weights, yaw, pitch, roll + d)
    grad_r = (r_r - r) / d    

    J[:, 0] = grad_y
    J[:, 1] = grad_p
    J[:, 2] = grad_r

    JTJ = J.T@J
    JTr = J.T@r
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
