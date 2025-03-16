import numpy as np
import cv2
import numba
import time

@numba.jit(nopython=True, parallel=True)
def convolve_numba(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Perform 2D convolution manually inside numba."""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    output = np.zeros((h, w), dtype=np.float64)
    
    # Manually pad the image (reflect padding)
    padded_image = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=np.float64)
    padded_image[pad_h:-pad_h, pad_w:-pad_w] = image
    padded_image[:pad_h, pad_w:-pad_w] = image[:pad_h, :]
    padded_image[-pad_h:, pad_w:-pad_w] = image[-pad_h:, :]
    padded_image[pad_h:-pad_h, :pad_w] = image[:, :pad_w]
    padded_image[pad_h:-pad_h, -pad_w:] = image[:, -pad_w:]
    
    for i in numba.prange(h):
        for j in range(w):
            region = padded_image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

@numba.jit(nopython=True, parallel=True)
def compute_energy_manual(image: np.ndarray) -> np.ndarray:
    """Compute the energy map of an image using a Sobel operator."""
    # Manually convert to grayscale (avoiding OpenCV)
    gray = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
    gray = gray.astype(np.float64)

    # Sobel kernels
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

    # Compute gradients manually
    Gx = convolve_numba(gray, Kx)
    Gy = convolve_numba(gray, Ky)

    return np.abs(Gx) + np.abs(Gy)

@numba.jit(nopython=True, parallel=True)
def find_seam(energy: np.ndarray) -> np.ndarray:
    """Find the lowest energy seam using dynamic programming."""
    h, w = energy.shape
    M = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int64)

    for i in range(1, h):
        for j in numba.prange(w):
            min_energy = M[i - 1, j]
            idx = j
            if j > 0 and M[i - 1, j - 1] < min_energy:
                min_energy = M[i - 1, j - 1]
                idx = j - 1
            if j < w - 1 and M[i - 1, j + 1] < min_energy:
                min_energy = M[i - 1, j + 1]
                idx = j + 1

            M[i, j] += min_energy
            backtrack[i, j] = idx

    seam = np.zeros(h, dtype=np.int64)
    seam[-1] = np.argmin(M[-1])

    for i in range(h - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]

    return seam

@numba.jit(nopython=True, parallel=True)
def remove_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Remove the lowest-energy seam from the image."""
    h, w, _ = image.shape
    new_image = np.zeros((h, w - 1, 3), dtype=np.uint8)

    for i in numba.prange(h):
        col = seam[i]
        # Manually copy pixels excluding the seam column
        new_image[i, :, :] = np.concatenate(
            (image[i, :col, :], image[i, col+1:, :]), axis=0
        )

    return new_image

def process_seam(image):
    """Compute energy, find seam, and remove it."""
    energy_map = compute_energy_manual(image)
    seam = find_seam(energy_map)
    return remove_seam(image, seam)

def seam_carving(image: np.ndarray, num_seams_vertical: int = 0) -> np.ndarray:
    """Remove `num_seams_vertical` seams using a loop."""
    for _ in range(num_seams_vertical):
        image = process_seam(image)

    return image

def main():
    t1 = time.time()
    
    # Load image
    image = cv2.imread("images/image.jpg")
    if image is None:
        raise ValueError("Error: Image not found. Check the file path.")

    num_seams_vertical = 400  # Adjust as needed

    resized_image = seam_carving(image, num_seams_vertical)

    cv2.imwrite("images/resized_image.jpg", resized_image)
    print("Seam carving complete. Image saved!")
    
    t2 = time.time()
    print("Time taken: ", t2 - t1)

if __name__ == "__main__":
    main()
