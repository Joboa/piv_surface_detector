# Free Surface Detection from PIV Images

Python algorithm for **automatic free-surface detection** in **raw PIV (Particle Image Velocimetry) images** based on image intensity.

The method is designed for grayscale PIV images where the free surface appears as a **bright, continuous horizontal feature** due to reflection and refraction effects.

---

## Overview of the Method

The algorithm detects the free surface using **intensity-based image processing**:

1. Normalize the input image for visualization
2. Apply strong horizontal smoothing to emphasize horizontal structures
3. Apply vertical Gaussian smoothing to reduce noise
4. Detect the brightest horizontal region column-by-column
5. Remove outliers and fit a smooth polynomial surface
6. Generate a binary mask to exclude the region above the free surface
7. (Optional) Visualize detection results for validation

---

## Input Data

- Grayscale PIV images (`uint16`, `uint8`, or normalized float)
- Typical camera data: **12-bit images stored as `uint16` (0–4095)**
- Bright seeding particles and reflective free surface

---

## Output

The function returns:

- `mask`  
  A binary mask (`uint8`) where:

  - White (255) = region to keep (below surface)
  - Black (0) = region to exclude (above surface)

- `y_line`  
  A 1D array containing the detected free surface position (row index) for each image column

---

## Usage

```python
from piv_surface_detector import detect_free_surface_intensity

mask, y_line = detect_free_surface_intensity(
    img,
    visualize=True
)
```

Set `visualize=True` to display:

- Original image
- Smoothed image used for detection
- Detected free surface overlay
- Binary mask
- Masked result
- Vertical intensity profile

## Dependencies

- Python 3.x
- NumPy
- OpenCV
- Matplotlib

Install dependencies with:

```python
pip install numpy opencv-python matplotlib
```

## Notes

- The algorithm converts images to 8-bit only for visualization and detection
- Raw image data should be preserved for PIV computation
- Parameters (kernel sizes, thresholds, buffers) can be adjusted depending on image resolution and flow conditions
