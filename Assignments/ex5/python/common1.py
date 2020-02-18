import numpy as np

# Task 1a
def central_difference(I):
    """
    Computes the gradient in the u and v direction using
    a central difference filter, and returns the resulting
    gradient images (Iu, Iv) and the gradient magnitude Im.
    """
    Iu = np.zeros_like(I)
    Iv = np.zeros_like(I)
    Im = np.zeros_like(I)

    difference_kernel = np.array([0.5, 0, -0.5])
    # Horizontal convolution
    for r in range(I.shape[0]):
        Iu[r, :] = np.convolve(I[r, :], difference_kernel, mode="same") 
    # Verticle convolution    
    for c in range(I.shape[1]):
        Iv[:, c] = np.convolve(I[:, c], difference_kernel.T, mode="same") 
    # Absolute magnitude
    Im = np.sqrt(Iv**2 + Iv**2)
    return Iu, Iv, Im

# Task 1b
def blur(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel, w, should depend on sigma, e.g.
    # w=2*np.ceil(3*sigma) + 1. Also, ensure that the blurred image
    # has the same size as the input image.
    kernel_len = 2 * np.ceil(3 * sigma) + 1
    x = np.linspace(-(kernel_len - 1) / 2., (kernel_len - 1) / 2., kernel_len)
    gaussian_kernel = np.exp(-x**2. / (2. * sigma**2)) / np.sqrt(2. * np.pi * sigma**2)
    
    # Horizontal convolution
    result = np.zeros_like(I)
    for r in range(I.shape[0]):
        result[r, :] = np.convolve(I[r, :], gaussian_kernel, mode="same")
    # Verticle convolution
    for c in  range(I.shape[1]):
        result[:, c] = np.convolve(result[:, c], gaussian_kernel.T, mode="same") 
    
    return result

# Task 1c
def extract_edges(Iu, Iv, Im, threshold):
    """
    Returns the u and v coordinates of pixels whose gradient
    magnitude is greater than the threshold.
    """

    # This is an acceptable solution for the task (you don't
    # need to do anything here). However, it results in thick
    # edges. If you want better results you can try to replace
    # this with a thinning algorithm as described in the text.
    v,u = np.nonzero(Im > threshold)
    theta = np.arctan2(Iv[v,u], Iu[v,u])
    return u, v, theta

def rgb2gray(I):
    """
    Converts a red-green-blue (RGB) image to grayscale brightness.
    """
    return 0.2989*I[:,:,0] + 0.5870*I[:,:,1] + 0.1140*I[:,:,2]
