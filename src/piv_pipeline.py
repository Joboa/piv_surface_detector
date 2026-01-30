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


def detect_free_surface_intensity_based(img, scan_width=True,
                                        x_start=None, x_end=None, auto_detect_bounds=True):
    """
    Detect free surface based on intensity: the surface is typically
    the brightest horizontal feature due to reflection/refraction

    Parameters:
    -----------
    img : ndarray
        Input image
    visualize : bool
        Whether to show diagnostic plots
    scan_width : bool
        Reserved for future use
    x_start : int, optional
        Starting x-coordinate for detection (default: auto-detect or 0)
    x_end : int, optional
        Ending x-coordinate for detection (default: auto-detect or image width)
    auto_detect_bounds : bool
        If True, automatically detect the valid horizontal region

    Returns:
    --------
    mask : ndarray
        Binary mask (255=keep, 0=remove)
    surface_line : ndarray
        Y-coordinates of detected surface
    x_coords : ndarray
        X-coordinates corresponding to surface_line
    metadata : dict
        Additional information (x_start, x_end, etc.)
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

    # Horizontal smoothing to find continuous bright regions
    kernel_size = (1, 15)
    smoothed = cv2.blur(img_normalized, kernel_size)

    # Vertical smoothing
    smoothed = cv2.GaussianBlur(smoothed, (5, 5), 0)

    # Auto-detect horizontal bounds if requested
    if auto_detect_bounds and x_start is None:
        # Look for columns with significant bright pixels in upper region
        search_height = int(height * 0.2)
        col_activity = np.max(smoothed[:search_height, :], axis=0)
        threshold = np.percentile(col_activity, 50)
        active_cols = np.where(col_activity > threshold)[0]

        if len(active_cols) > 0:
            x_start = active_cols[0]
            # x_end = active_cols[-1] + 1 if x_end is None else x_end
            x_end = width
            print(f"Auto-detected bounds: x_start={x_start}, x_end={x_end}")
        else:
            x_start = 0
            x_end = width
            print("Could not auto-detect bounds, using full width")

    # Use defaults if not set
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = width

    # Ensure valid bounds
    x_start = max(0, x_start)
    x_end = min(width, x_end)

    print(f"Detection region: x=[{x_start}, {x_end}), width={x_end - x_start}")

    # Search region
    surface_points = []
    search_height = int(height * 0.2)

    if scan_width:
        pass

    # Only scan the active region
    for col in range(x_start, x_end, 2):
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
        row_averages = np.mean(smoothed[:search_height, x_start:x_end], axis=1)
        surface_row = np.argmax(row_averages)

        # Create a flat horizontal line for the active region
        x_line = np.arange(x_start, x_end)
        y_line = np.full(x_end - x_start, surface_row)

    else:
        print(f"Using {len(surface_points)} computed points")
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
            print("Not enough points after filtering. Using median...")
            y_line = np.full(x_end - x_start, y_median)
            x_line = np.arange(x_start, x_end)
        else:
            sort_idx = np.argsort(x_filtered)
            x_sorted = x_filtered[sort_idx]
            y_sorted = y_filtered[sort_idx]

            # Polynomial fitting
            degree = min(6, len(x_sorted) - 1)
            coeffs = np.polyfit(x_sorted, y_sorted, deg=degree)
            poly = np.poly1d(coeffs)

            x_line = np.arange(x_start, x_end)
            y_line = poly(x_line)

            # Within search region
            y_line = np.clip(y_line, 0, search_height)

    # Mask for the free surface
    # mask = np.ones_like(img, dtype=np.uint8) * 255
    mask = np.zeros_like(img, dtype=np.uint8)

    # Buffer above and below detected line
    buffer_above = 10
    buffer_below = 20

    # Apply mask only in the detected region
    for i, col in enumerate(x_line):
        if 0 <= col < width and i < len(y_line):
            surface_row = int(y_line[i])
            start_row = max(0, surface_row - buffer_above)
            end_row = min(height, surface_row + buffer_below)
            # mask[:end_row, col] = 0  # Mask from top to below surface
            mask[:end_row, col] = 255  # Mask from top to below surface

    # Metadata for alignment with PIV fields
    metadata = {
        'x_start': x_start,
        'x_end': x_end,
        'x_coords': x_line,
        'mean_surface_row': np.mean(y_line),
        'surface_std': np.std(y_line),
        'n_points': len(surface_points)
    }

    return mask, y_line, x_line, metadata


def process_piv_image_pair(la_path, lb_path, output_dir, visualize_first=False, use_PNG=False, save_mat=False):
    """
    Process a pair of PIV images (LA and LB) and save masked versions
    """
    import uuid

    # Read both images
    img_la = mpimg.imread(la_path)
    img_lb = mpimg.imread(lb_path)

    # Detect surface from LA image
    mask, surface_line, x_line, metadata = detect_free_surface_intensity_based(
        img_la)

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

    # ROI cell data if save_mat is True
    roi_cell = None
    if save_mat:
        # Boundary contour points from mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Get the largest contour (the main mask region)
            largest_contour = max(contours, key=cv2.contourArea)
            # Convert to nx2 array (x, y coordinates)
            boundary_points = largest_contour.reshape(-1, 2).astype(np.float64)
            boundary_points = boundary_points + 1  # MATLAB 1-based indexing
        else:
            # Fallback: use surface line points
            print(f"Using surface line points")
            boundary_points = np.column_stack(
                (x_line, surface_line)).astype(np.float64)
            boundary_points = boundary_points + 1  # MATLAB 1-based indexing

        # Unique ID for this mask
        unique_id = 'tp' + str(uuid.uuid4()).replace('-', '_')

        # PIVLab format structure with 5 elements
        # Structure:
        #   {1,1}: 'ROI_object_external'
        #   {1,2}: nx2 boundary points
        #   {1,3}: [0.2422, 0.1504, 0.6603] - RGB color
        #   {1,4}: [] - empty array
        #   {1,5}: unique ID string
        roi_cell = np.empty((1, 5), dtype=object)
        roi_cell[0, 0] = 'ROI_object_external'
        roi_cell[0, 1] = boundary_points
        roi_cell[0, 2] = np.array([0.2422, 0.1504, 0.6603])  # RGB color
        roi_cell[0, 3] = ""  # np.array([])  # Empty array
        roi_cell[0, 4] = unique_id  # Unique ID

        print(
            f"PIVLab ROI cell with {len(boundary_points)} boundary points, ID: {unique_id}")

    elif use_PNG:
        # PNG compression on binary data
        output_la = output_la.replace('.TIF', '_mask.png')
        output_lb = output_lb.replace('.TIF', '_mask.png')

        # PNG compression
        cv2.imwrite(output_la, mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(output_lb, mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    else:
        # Default: save as regular mask image
        cv2.imwrite(output_la, mask)
        cv2.imwrite(output_lb, mask)

    # File size comparison
    if not save_mat:
        original_size = mask.size * mask.itemsize  # bytes
        import os as os_module

        # Check the actual output file
        if use_PNG:
            check_file = output_la.replace('.TIF', '_mask.png')
        else:
            check_file = output_la

        if os_module.path.exists(check_file):
            compressed_size = os_module.path.getsize(check_file)
            print(
                f"File size: {original_size} bytes → {compressed_size} bytes ({compressed_size/original_size*100:.1f}%)")

    return mask, surface_line, masked_la, masked_lb, metadata, roi_cell


def batch_process_piv_images(input_dir, output_dir, pattern="*.TIF", visualize_first=True, use_PNG=False, save_mat=False):
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
    use_PNG : bool
        If True, save masks as compressed PNG files
    save_mat : bool
        If True, save all masks in single PIVLab .mat format file
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

    # All ROI cells for .mat format
    all_roi_cells = []

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
            mask, surface_line, masked_la, masked_lb, metadata, roi_cell = process_piv_image_pair(
                la_path, lb_path, output_dir, use_PNG=use_PNG, save_mat=save_mat
            )

            success_count += 1
            surface_positions.append(np.mean(surface_line))

            if save_mat and roi_cell is not None:
                all_roi_cells.append(roi_cell)

            # Visualize first pair
            if visualize_first and idx == 0:
                visualize_processing_result(
                    mpimg.imread(la_path),
                    masked_la,
                    mask,
                    surface_line,
                    os.path.basename(la_path),
                    x_coords=metadata['x_coords'],
                )

        except Exception as e:
            print(f"\nError processing {os.path.basename(la_path)}: {str(e)}")
            fail_count += 1

    # Save all masks in single .mat file for PIVLab
    if save_mat and len(all_roi_cells) > 0:
        from scipy.io import savemat

        # 1 x n cell array containing all masks
        masks_in_frame = np.empty((1, len(all_roi_cells)), dtype=object)
        for i, roi_cell in enumerate(all_roi_cells):
            masks_in_frame[0, i] = roi_cell

        # Save to single .mat file
        mat_output_path = os.path.join(output_dir, 'masks_in_frame.mat')
        savemat(mat_output_path, {
                'masks_in_frame': masks_in_frame}, format='5')
        print(f"\nSaved {len(all_roi_cells)} masks to {mat_output_path}")
        print(f"  Structure: masks_in_frame [1×{len(all_roi_cells)} cell]")

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


def visualize_processing_result(original, masked, mask, surface_line, filename, x_coords):
    """
    Visualize the processing result for quality checking
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmin, vmax = np.percentile(original, (1, 99))

    # Original with detected surface
    axes[0].imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].plot(x_coords, surface_line, 'r-', linewidth=2)
    axes[0].set_title(f"Original + Detected Surface\n{filename}")
    axes[0].axis("off")

    # Mask
    axes[1].imshow(mask, cmap="gray")  # masked
    axes[1].plot(x_coords, surface_line, 'r-', linewidth=1)
    # axes[1].set_title("Mask\n(white=keep, black=remove)")
    axes[1].set_title("Mask\n(black=keep, white=remove)")
    axes[1].axis("off")

    # Masked result
    axes[2].imshow(masked, cmap="gray", vmin=vmin, vmax=vmax)  # masked_la
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
            mask, surface_line, masked_la, masked_lb, metadata = process_piv_image_pair(
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
                    os.path.basename(la_path),
                    x_coords=metadata['x_coords'],
                )

        except Exception as e:
            print(f"\nError: {str(e)}")
