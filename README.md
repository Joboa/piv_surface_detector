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
- LA and LB image pairs (optional, for full PIV sequence processing)

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

For a quick check with one PIV image (either LA or LB):

```python
from piv_surface_detector import detect_free_surface_intensity

mask, y_line = detect_free_surface_intensity(
    img,
    visualize=True
)
```

For sequential PIV data processing (with both LA and LB):

```python
from src.piv_pipeline import batch_process_piv_images

batch_process_piv_images(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        visualize_first=True
    )
```

Set `visualize=True` to display:

- Original image
- Smoothed image used for detection
- Detected free surface overlay
- Binary mask
- Masked result
- Vertical intensity profile

## Batch process a folder of PIV images

Use `main.py` as an entry point:

```python
python main.py
```

By default, it will:

- Process all images in `./raw_piv`
- Save masked outputs to `./masked_images`
- Visuallize the first pair for quick inspection

You can also process with periodice quality checks by uncommenting the relevant section in `main.py`

## Proces LA & LB image pairs

The functions in `src/piv_pipeline.py` can handle full PIV sequences, including LA and LB pairs. This allows consistent detection and masking across image pairs.

## File Structure

```python
├── src/
│   └── piv_pipeline.py          # Core functions: detection, batch processing, visualization
├── tutorial/
├── main.py                      # Entry point for batch processing and quality checks
├── piv_surface_detector.py      # For quick check with one image
├── README.md
├── raw_image/                   # Example image
├── masked_images/               # Output folder for masks (created automatically)
└── .gitignore
```

## Dependencies

- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- Tqdm

Install dependencies with:

```python
pip install numpy opencv-python matplotlib tqdm
```

## Notes

- The algorithm converts images to 8-bit only for visualization and detection
- Raw image data should be preserved for PIV computation
- Parameters (kernel sizes, thresholds, buffers) can be adjusted depending on image resolution and flow conditions

## Limitations

While this algorithm provides robust free-surface detection for PIV images, it has some limitations:

1. Assumption of brightness:
The method assumes the free surface is the brightest horizontal feature within the search region.

2. Sensitivity to foam or bubbles:
Excessive surface foam, bubbles or reflections can interfere with detection and may reduce accuracy.

3. Manual adjustment required:
The search region and other parameters may need manual tuning for different flow conditins or camera setups.

## Very IMPORTANT

Users should visually inspect results when processing new datasets, especially for flows with high turbulence or surface disturbances, and adjust parameters accordingly.
