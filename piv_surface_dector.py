"""
Free Surface Detection Algorithm for PIV Images

Author: John Boamah
Year: 2025

This code is part of ongoing academic research.
Redistribution, modification, or commercial use is not permitted
without explicit permission from the author.
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def detect_free_surface_intensity(img, visualize=False):

    # Image normalization
    if img.dtype == np.uint16:
        print(f"Using image type of {img.dtype}")
        vmin, vmax = np.percentile(img, (1, 99))
        img_normalized = np.clip(img, vmin, vmax)
        img_normalized = ((img_normalized - vmin) /
                          (vmax - vmin) * 255).astype(np.uint8)
    elif img.dtype == np.uint8:
        print(f"Using image type of {img.dtype}")
        img_normalized = img.copy()
    elif np.issubdtype(img.dtype, np.floating):
        print(f"Using image type of {img.dtype}")
        img_normalized = np.clip(img, 0.0, 1.0)
        img_normalized = (img_normalized * 255).astype(np.uint8)
    print(
        f"Image (Actual) size: max image: {img.max()}, min image: {img.min()}")
    print(
        f"Image (Normalized) size: max image: {img_normalized.max()}, min image: {img_normalized.min()}")

    print(f"Image dimensions: {img.shape}")
    height, width = img.shape[:2]

    # Horizontal smothing to find continous bright regions
    kernel_size = (1, 15)
    smoothed = cv2.blur(img_normalized, kernel_size)

    # Vertical smoothing
    smoothed = cv2.GaussianBlur(smoothed, (5, 5), 0)

    # Search region
    surface_points = []
    search_height = int(height * 0.2)

    for col in range(0, width, 2):
        column = smoothed[:search_height, col]
        window_size = 15
        max_avg = 0  # max average intensity
        max_pos = 0  # the position (row) it occurs

        for row in range(window_size, search_height - window_size):
            # how bright is the window around this area
            window_avg = np.mean(column[row-window_size:row+window_size])
            if window_avg > max_avg:
                max_avg = window_avg
                max_pos = row

        threshold = np.percentile(column, 70)  # 30% of the brightest pixels
        if max_avg > threshold:
            # column and position of the maximum intensity
            surface_points.append((col, max_pos))

    if len(surface_points) < 10:
        print("Warning: Very few surface points detected")
        row_averages = np.mean(smoothed[:search_height, :], axis=1)
        surface_row = np.argmax(row_averages)

        # Create a flat horizontal line
        x_line = np.arrange(0, width)
        y_line = np.full(width, surface_row)

    else:
        print(f"Using the computed points")
        x_points = np.array([p[0] for p in surface_points])  # columns
        # position of the various intensities
        y_points = np.array([p[1] for p in surface_points])

        # Outlier removal using percentile-based method
        y_median = np.median(y_points)
        y_std = np.std(y_points)

        # Points to keep (within 2 standard deviations)
        mask_inliers = np.abs(y_points - y_median) < (2 * y_std)
        x_filtered = x_points[mask_inliers]
        y_filtered = y_points[mask_inliers]

        if len(x_filtered) < 5:
            print("Not enough points. Using median...")
            y_line = np.full(width, y_median)
            x_line = np.arange(0, width)
        else:
            sort_dix = np.argsort(x_filtered)
            x_sorted = x_filtered[sort_dix]
            y_sorted = y_filtered[sort_dix]

            # Polynomial fitting
            degree = min(3, len(x_sorted) - 1)
            coeffs = np.polyfit(x_sorted, y_sorted, deg=degree)
            poly = np.poly1d(coeffs)

            x_line = np.arange(0, width)
            y_line = poly(x_line)

            # Within search region
            y_line = np.clip(y_line, 0, search_height)

    # Mask for the free surface
    mask = np.ones_like(img, dtype=np.uint8) * 255

    # Buffer above and below detected line
    buffer_above = 10
    buffer_below = 20

    for col in range(width):
        surface_row = int(y_line[col] if col < len(y_line) else y_line[-1])
        start_row = max(0, surface_row - buffer_above)
        end_row = min(height, surface_row + buffer_below)
        mask[:end_row, col] = 0  # Mask from top to below surface

    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        vmin, vmax = np.percentile(img, (1, 99))

        # Original image
        axes[0, 0].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Smoothed image
        axes[0, 1].imshow(smoothed, cmap="gray")
        axes[0, 1].set_title("Smoothed (for detection)")
        axes[0, 1].axis("off")

        # Detected surface overlay
        axes[0, 2].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, 2].plot(x_line, y_line, 'r-', linewidth=2,
                        label='Detected Surface')
        if len(surface_points) > 0:
            axes[0, 2].scatter([p[0] for p in surface_points],
                               [p[1] for p in surface_points],
                               c='yellow', s=3, alpha=0.6, label='Sample Points')
        axes[0, 2].set_title("Detected Free Surface")
        axes[0, 2].legend()
        axes[0, 2].axis("off")

        # Mask
        axes[1, 0].imshow(mask, cmap="gray")
        axes[1, 0].plot(x_line, y_line, 'r-', linewidth=1)
        axes[1, 0].set_title("Mask (white=keep, black=remove)")
        axes[1, 0].axis("off")

        # Masked result
        masked_img = img.copy()
        masked_img[mask == 0] = 0
        axes[1, 1].imshow(masked_img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1, 1].set_title("Masked Result")
        axes[1, 1].axis("off")

        # Intensity profile (average per row)
        row_intensities = np.mean(smoothed[:int(height*0.5), :], axis=1)
        axes[1, 2].plot(row_intensities, range(len(row_intensities)))
        axes[1, 2].axhline(y=np.mean(y_line), color='r', linestyle='--',
                           label=f'Detected surface (row {int(np.mean(y_line))})')
        axes[1, 2].set_xlabel('Average Intensity')
        axes[1, 2].set_ylabel('Row (from top)')
        axes[1, 2].set_title('Vertical Intensity Profile')
        axes[1, 2].invert_yaxis()
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return mask, y_line

    return mask, y_line


# Test
if __name__ == "__main__":
    img = mpimg.imread("raw_image/SCC0.LA.TIF")
    mask, surface_line = detect_free_surface_intensity(img, visualize=True)

    if mask is not None:
        print(f"Surface detected at average row: {np.mean(surface_line):.1f}")
        print(f"Surface variation (std): {np.std(surface_line):.1f} pixels")
