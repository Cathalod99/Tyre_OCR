import math
import numpy as np
import cv2

def warpPolar(path):
    ring = cv2.imread(path)
    size = max(ring.shape[0],  ring.shape[1])
    width = ring.shape[1]
    height = ring.shape[0]
    # All points are in format [cols, rows]
    src_points = np.float32([[0, 0], [width - 1, 0], [width -  1, height - 1], [0, height - 1]])
    dst_points = np.float32([[0, 0],  [size - 1, 0],  [size - 1, size - 1],  [0, size - 1]]) 
    
    M = cv2.getPerspectiveTransform(src_points, dst_points) 
    ring = cv2.warpPerspective(ring, M, (size, size))

    cv2.imwrite(f"{path.split('.')[0]}_tyre_square.jpg", ring)

    outer_radius = size // 2
    inner_radius_factor = 0.6  # 0.70 measured empirically from image

    # Unwarp ring
    warped = cv2.warpPolar(ring, (size, int(size * math.pi)), (outer_radius, outer_radius), outer_radius, 0)

    # Rotate 90 degrees
    straightened = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Crop to ring only
    cropped = straightened[: int(straightened.shape[0] * (1 - inner_radius_factor)), :]
    combined_image = cv2.hconcat([cropped, cropped]) 
    cv2.imwrite(f"{path.split('.')[0]}_convert.jpg", combined_image)