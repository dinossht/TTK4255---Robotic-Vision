import matplotlib.pyplot as plt
import numpy as np

def camera_to_fisheye(X, Y, Z, intrinsics):
    """
    X,Y,Z:      3D point in camera coordinates. Z-axis pointing forward,
                Y-axis pointing down, and X-axis point to the right.
    intrinsics: Fisheye intrinsics [f, cx, cy, k]
    """
    f,cx,cy,k = intrinsics
    theta = np.arctan2(np.sqrt(X*X + Y*Y), Z)
    phi = np.arctan2(Y, X)
    r = f*theta*(1 + k*theta*theta)
    u = cx + r*np.cos(phi)
    v = cy + r*np.sin(phi)
    return [u, v]


#
# Test code: Draw a grid of points
#

intrinsics = np.loadtxt('data/intrinsics.txt', comments='%')
X = np.linspace(-2,+2,10)
Y = np.linspace(-1,+1,5)
uv = []
for X_i in X:
    for Y_i in Y:
        Z_i = 2.0
        uv.append(camera_to_fisheye(X_i, Y_i, Z_i, intrinsics))
uv = np.array(uv)
plt.scatter(uv[:,0], uv[:,1])
plt.axis('scaled')
plt.xlim([0, 1280])
plt.ylim([720, 0])
plt.grid()
plt.show()
