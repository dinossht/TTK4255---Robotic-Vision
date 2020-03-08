import numpy as np

def epipolar_match(I1, I2, F, uv1):
    """
    For each point in uv1, finds the matching point in image 2 by
    an epipolar line search.

    Args:
        I1:  (H x W matrix) Grayscale image 1
        I2:  (H x W matrix) Grayscale image 2
        F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1 to lines in image 2
        uv1: (n x 2 array) Points in image 1

    Returns:
        uv2: (n x 2 array) Best matching points in image 2.
    """

    # Tips:
    # - Image indices must always be integer.
    # - Use int(x) to convert x to an integer.
    # - Use rgb2gray to convert images to grayscale.
    # - Skip points that would result in an invalid access.
    # - Use I[v-w : v+w+1, u-w : u+w+1] to extract a window of half-width w around (v,u).
    # - Use the np.sum function.

    # Start iterasjon over alle punkter i uv1
    # for hvert punkt: regn ut linje i andre bilde
    # for hvert punkt nedover den linjen: regn ut hvor "likt" det er rundt punktet på linjen med punktet i det andre bildet (husk, et punkt er blitt en linje)
    # velge punkt i andre bilde som "ligner" mest på punkt i første bilde, putt inn i matrise som er like stor som uv1

    n = uv1.shape[0]
    H, W = I1.shape
    w = 10  # image comparison image size
    
    uv2 = np.zeros(uv1.shape)
    # Calculate epipolar line in image 2 for a given point in image 1 
    l = (F @ np.block([uv1, np.ones((n, 1))]).T).T
    
    best_SAD = np.inf
    
    for i in range(len(uv1)):
        for u in range(w, H):
            v = int((1.0 / l[i][1]) * (-l[i][2] - u * l[i][0]))
            
            # Window start-end indices
            u2_start    = u - w
            u2_end      = u + w + 1
            v2_start    = v - w
            v2_end      = v + w + 1
            u1_start    = int(uv1[i, 0] - w)
            u1_end      = int(uv1[i, 0] + w + 1)
            v1_start    = int(uv1[i, 1] - w)
            v1_end      = int(uv1[i, 1] + w + 1)

            # Check if indices are inside the image
            if (u2_start >= 0 and u2_end < W and v2_start >= 0 and v2_end < H \
                and u1_start >= 0 and u1_end < W and v1_start >= 0 and v1_end < H):
                I_ref = I1[v1_start:v1_end, u1_start:u1_end]
                I_comp = I2[v2_start:v2_end, u2_start:u2_end]

                # Calculate best sum of absolute intensity difference (SAD)
                current_SAD = np.sum((I_ref - I_comp)**2)
                if best_SAD >= current_SAD:
                    best_SAD = current_SAD
                    uv2[i, 0] = u
                    uv2[i, 1] = v

    return uv2
