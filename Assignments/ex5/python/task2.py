import numpy as np
import matplotlib.pyplot as plt
from common1 import *
from common2 import *

edge_threshold = 0 # todo: choose an appropriate value
blur_sigma     = 0 # todo: choose an appropriate value
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
bins      = 100 # Placeholder
rho_max   = 0 # Placeholder
rho_min   = 0 # Placeholder
theta_min = 0 # Placeholder
theta_max = 0 # Placeholder
H = np.zeros([bins,bins]) # Placeholder

# Tip: Use histogram2d for task 2a
# H, _, _ = np.histogram2d(theta, rho, bins=bins, range=[[theta_min, theta_max], [rho_min, rho_max]])
# H = H.T # Make rows be rho and columns be theta (see documentation)

#
# Task 2b: Find local maxima
#
line_threshold = 15 # Placeholder
window_size = 10 # Placeholder
peak_rows,peak_cols = extract_peaks(H, window_size, line_threshold)

#
# Task 2c: Convert peak (row, column) pairs into (theta, rho) pairs.
#
peak_theta = [0, 0.2, 0.5, 0.7] # Placeholder to demonstrate use of draw_line
peak_rho   = [10, 100, 300, 500] # Placeholder to demonstrate use of draw_line

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
# plt.show()
