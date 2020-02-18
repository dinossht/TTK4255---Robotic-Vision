import numpy as np
import matplotlib.pyplot as plt
from common1 import *
from common2 import *

edge_threshold = 0.01 
blur_sigma     = 1 
filename       = '../data/image1_und.jpg'

I_rgb      = plt.imread(filename)
I_rgb      = I_rgb/255.0
I_gray     = rgb2gray(I_rgb)
I_blur     = blur(I_gray, blur_sigma)
Iu, Iv, Im = central_difference(I_blur)
u,v,theta  = extract_edges(Iu, Iv, Im, edge_threshold)

#
# Task 2a: Compute accumulator array H
#
bins      = 100 
rho_max   = 873 # diagonal length of image
rho_min   = 0
theta_max = 2 * np.pi # Placeholder
theta_min = 0 
#H = np.zeros([bins,bins])

# Tip: Use histogram2d for task 2a
rho = np.round(u * np.cos(theta) + v * np.sin(theta))
H, _, _ = np.histogram2d(theta, rho, bins=bins, range=[[theta_min, theta_max], [rho_min, rho_max]])
H = H.T # Make rows be rho and columns be theta (see documentation)

#
# Task 2b: Find local maxima
#
line_threshold = 30 
window_size = 20 
peak_rows,peak_cols = extract_peaks(H, window_size, line_threshold)

#
# Task 2c: Convert peak (row, column) pairs into (theta, rho) pairs.
#
peak_theta = (theta_min + (theta_max - theta_min) * peak_cols) / bins
peak_rho   = (rho_min + (rho_max - rho_min) * peak_rows) / bins

plt.figure(figsize=[6,8])
plt.subplot(211)
plt.imshow(H, extent=[theta_min, theta_max, rho_min, rho_max], aspect='auto')
plt.xlabel('$\\theta$ (radians)')
plt.ylabel('$\\rho$ (pixels)')
plt.colorbar(label='Votes')
plt.title('Hough transform histogram')
plt.subplot(212)
plt.imshow(I_rgb)
plt.xlim([0, I_rgb.shape[1]])
plt.ylim([I_rgb.shape[0], 0])
for i in range(len(peak_theta)):
    draw_line(peak_theta[i], peak_rho[i], color='yellow')
plt.tight_layout()
plt.savefig('out2.png')
plt.show()
