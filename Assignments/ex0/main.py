import matplotlib.pyplot as  plt
import numpy as np


# Load image
img = plt.imread("roomba.jpg")

# Extract RGB
r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]

print(r)
