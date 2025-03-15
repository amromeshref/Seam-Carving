import numpy as np
import cv2

def compute_energy_manual(image: np.ndarray) -> np.ndarray:
    """
    Compute the energy map of an image using the absolute gradient in x and y directions.
    Input:
        image: numpy array. BGR Format.
    Returns:
        energy: numpy array of shape (H x W) representing the energy map
    """
    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
    
    h, w = gray.shape
    energy = np.zeros((h, w), dtype=np.float64)  # Initialize with 0

    # Compute gradients manually
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            dx = abs(int(gray[i, j+1]) - int(gray[i, j-1]))  # Gradient in x-direction
            dy = abs(int(gray[i+1, j]) - int(gray[i-1, j]))  # Gradient in y-direction
            energy[i, j] = dx + dy  # Given energy function

    # Handle edges by mirroring nearest values
    energy[0, :] = energy[1, :]
    energy[-1, :] = energy[-2, :]
    energy[:, 0] = energy[:, 1]
    energy[:, -1] = energy[:, -2]

    return energy

def find_seam(energy: np.ndarray) -> np.ndarray:
    """
    Find the seam with the lowest energy using dynamic programming.
    Input:
        energy: numpy array of shape (H x W) representing the energy map
    Returns:
        seam: numpy array of shape (H,) containing the indices of the lowest-energy seam
    """
    h, w = energy.shape
    M = np.copy(energy)
    backtrack = np.zeros_like(M, dtype=np.int64)

    # Compute the cumulative energy map
    for i in range(1, h):
        for j in range(w):
            min_energy = M[i-1, j]
            idx = j
            if j > 0 and M[i-1, j-1] < min_energy:
                min_energy = M[i-1, j-1]
                idx = j-1
            if j < w-1 and M[i-1, j+1] < min_energy:
                min_energy = M[i-1, j+1]
                idx = j+1

            M[i, j] += min_energy
            backtrack[i, j] = idx

    # Find the lowest energy seam
    seam = np.zeros(h, dtype=np.int64)
    seam[-1] = np.argmin(M[-1])  # Start from last row

    for i in range(h-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]

    return seam

def remove_seam(image: np.ndarray, seam: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    """
    Remove the lowest-energy seam from the image.
    Input:
        image: numpy array of shape (H x W x 3) representing the image
        seam: numpy array of shape (H,) containing the indices of the seam to be removed
        seam_mask: numpy array of shape (H x W x 3) representing the seam visualization mask
    Returns:
        new_image: numpy array of shape (H x (W-1) x 3) representing the image with seam removed
    """
    h, w, _ = image.shape
    new_image = np.zeros((h, w-1, 3), dtype=np.uint8)

    for i in range(h):
        col = seam[i]
        new_image[i, :, :] = np.delete(image[i, :, :], col, axis=0)  # Use axis=1 for width removal
        seam_mask[i, col] = [0, 0, 255]  # Mark seam in red

    return new_image

def seam_carving(image: np.ndarray, num_seams_vertical:int=0, num_seams_horizontal:int=0) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove `num_seams_vertical` vertical and `num_seams_horizontal` horizontal seams using seam carving.
    Input:
        image: numpy array of shape (H x W x 3) representing the image
        num_seams_vertical: int. Number of vertical seams to remove
        num_seams_horizontal: int. Number of horizontal seams to remove
    Returns:
        image: numpy array representing the resized image
        seam_mask: numpy array representing the seam visualization mask
    """
    seam_mask = np.copy(image)  # Create a mask to visualize seams

    # Remove vertical seams
    for _ in range(num_seams_vertical):
        energy_map = compute_energy_manual(image)
        seam = find_seam(energy_map)
        image = remove_seam(image, seam, seam_mask)

    # Rotate image 90 degrees to remove horizontal seams
    if num_seams_horizontal > 0:
        image = np.rot90(image, 1, (0, 1))
        seam_mask = np.rot90(seam_mask, 1, (0, 1))

        # Remove horizontal seams (by rotating image)
        for _ in range(num_seams_horizontal):
            energy_map = compute_energy_manual(image)
            seam = find_seam(energy_map)
            image = remove_seam(image, seam, seam_mask)

        # Rotate image back to original orientation
        image = np.rot90(image, -1, (0, 1))
        seam_mask = np.rot90(seam_mask, -1, (0, 1))

    return image, seam_mask


def main():
    # Load image
    image = cv2.imread("images/image.jpg")
    if image is None:
        raise ValueError("Error: Image not found. Check the file path.")

    # Reduce the size of the image
    num_seams_vertical = 400 # Reduce width
    num_seams_horizontal = 200 # Reduce height
    resized_image, seam_visualization = seam_carving(image, num_seams_vertical, num_seams_horizontal)

    # Save output images
    cv2.imwrite("images/resized_image.jpg", resized_image)
    cv2.imwrite("images/seam_visualization.jpg", seam_visualization)

    print("Seam carving complete. Images saved!")


if __name__ == "__main__":
    main()