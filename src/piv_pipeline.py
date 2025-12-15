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
import os
import glob
from tqdm import tqdm


def detect_free_surface_intensity_based(img):
    """
    Detect free surface based on intensity: the surface is typically
    the brightest horizontal feature due to reflection/refraction
    """

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

    return mask, y_line


def process_piv_image_pair(la_path, lb_path, output_dir, visualize_first=False):
    """
    Process a pair of PIV images (LA and LB) and save masked versions
    """
    # Read both images
    img_la = mpimg.imread(la_path)
    img_lb = mpimg.imread(lb_path)

    # Detect surface from LA image
    mask, surface_line = detect_free_surface_intensity_based(img_la)

    # Apply same mask to both images
    masked_la = img_la.copy()
    masked_lb = img_lb.copy()

    masked_la[mask == 0] = 0
    masked_lb[mask == 0] = 0

    # Output filenames
    la_basename = os.path.basename(la_path)
    lb_basename = os.path.basename(lb_path)

    output_la = os.path.join(output_dir, la_basename)
    output_lb = os.path.join(output_dir, lb_basename)

    # Save masked images
    cv2.imwrite(output_la, masked_la)
    cv2.imwrite(output_lb, masked_lb)

    return mask, surface_line, masked_la, masked_lb


def batch_process_piv_images(input_dir, output_dir, pattern="*.TIF", visualize_first=True):
    """
    Batch process all PIV image pairs in a directory

    Parameters:
    -----------
    input_dir : str
        Directory containing input TIF files
    output_dir : str
        Directory to save masked images
    pattern : str
        File pattern to match (default: "*.TIF")
    visualize_first : bool
        Whether to show visualization for the first pair
    """
    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all LA files (first frame of each pair)
    la_files = sorted(glob.glob(os.path.join(input_dir, "*LA.TIF")))

    if len(la_files) == 0:
        print(f"No LA.TIF files found in {input_dir}")
        return

    print(f"Found {len(la_files)} image pairs to process")

    # Statistics tracking
    success_count = 0
    fail_count = 0
    surface_positions = []

    # Process each pair
    for idx, la_path in enumerate(tqdm(la_files, desc="Processing PIV pairs")):
        # Corresponding LB path
        lb_path = la_path.replace(".LA.TIF", ".LB.TIF")

        if not os.path.exists(lb_path):
            print(
                f"\nWarning: Missing LB file for {os.path.basename(la_path)}")
            fail_count += 1
            continue

        try:
            # Process the pair
            mask, surface_line, masked_la, masked_lb = process_piv_image_pair(
                la_path, lb_path, output_dir
            )

            success_count += 1
            surface_positions.append(np.mean(surface_line))

            # Visualize first pair
            if visualize_first and idx == 0:
                visualize_processing_result(
                    mpimg.imread(la_path),
                    masked_la,
                    mask,
                    surface_line,
                    os.path.basename(la_path)
                )

        except Exception as e:
            print(f"\nError processing {os.path.basename(la_path)}: {str(e)}")
            fail_count += 1

    # Process summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total pairs found:       {len(la_files)}")
    print(f"Successfully processed:  {success_count}")
    print(f"Failed:                  {fail_count}")

    if len(surface_positions) > 0:
        print(f"\nFree Surface Statistics:")
        print(
            f"  Mean position:         {np.mean(surface_positions):.1f} pixels from top")
        print(
            f"  Std deviation:         {np.std(surface_positions):.1f} pixels")
        print(
            f"  Min position:          {np.min(surface_positions):.1f} pixels")
        print(
            f"  Max position:          {np.max(surface_positions):.1f} pixels")
        print(
            f"  Range:                 {np.max(surface_positions) - np.min(surface_positions):.1f} pixels")

    print(f"\nMasked images saved to: {output_dir}")
    print("="*60)


def visualize_processing_result(original, masked, mask, surface_line, filename):
    """
    Visualize the processing result for quality checking
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmin, vmax = np.percentile(original, (1, 99))

    # Original with detected surface
    axes[0].imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].plot(range(len(surface_line)), surface_line, 'r-', linewidth=2)
    axes[0].set_title(f"Original + Detected Surface\n{filename}")
    axes[0].axis("off")

    # Mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].plot(range(len(surface_line)), surface_line, 'r-', linewidth=1)
    axes[1].set_title("Mask\n(white=keep, black=remove)")
    axes[1].axis("off")

    # Masked result
    axes[2].imshow(masked, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title("Masked Result")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def process_with_quality_check(input_dir, output_dir, check_interval=50):
    """
    Process images with periodic quality checks

    Parameters:
    -----------
    input_dir : str
        Directory containing input TIF files
    output_dir : str
        Directory to save masked images
    check_interval : int
        Show visualization every N images for quality checking
    """
    os.makedirs(output_dir, exist_ok=True)

    la_files = sorted(glob.glob(os.path.join(input_dir, "*LA.TIF")))

    print(f"Found {len(la_files)} image pairs")
    print(f"Will show quality check every {check_interval} images")

    for idx, la_path in enumerate(tqdm(la_files, desc="Processing")):
        lb_path = la_path.replace(".LA.TIF", ".LB.TIF")

        if not os.path.exists(lb_path):
            continue

        try:
            mask, surface_line, masked_la, masked_lb = process_piv_image_pair(
                la_path, lb_path, output_dir
            )

            # Quality check at intervals
            if idx % check_interval == 0:
                print(f"\nQuality check for image {idx+1}/{len(la_files)}")
                visualize_processing_result(
                    mpimg.imread(la_path),
                    masked_la,
                    mask,
                    surface_line,
                    os.path.basename(la_path)
                )

        except Exception as e:
            print(f"\nError: {str(e)}")
