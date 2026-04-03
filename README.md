# Simple_feature_extraction


A Python script that extracts and visualizes key visual features from an image using OpenCV.

## Features

| Technique | Description |
|---|---|
| **Canny Edges** | Detects sharp intensity boundaries |
| **Harris Corners** | Finds corner points using gradient analysis |
| **ORB Keypoints** | Scale/rotation-invariant keypoints + binary descriptors |
| **Blob Detection** | Locates circular or uniform regions |
| **Contour Detection** | Extracts object boundaries |
| **HOG Descriptor** | Histogram of oriented gradients for shape representation |
| **Color Histogram** | Pixel intensity distribution per RGB channel |

All results are displayed in a single dashboard and saved as a PNG file.

## Requirements

```bash
pip install opencv-python numpy matplotlib
```

## Usage

1. Clone or download the script.
2. Set your image path in the main block:
```python
IMAGE_PATH = "your_image.jpg"
```
3. Run the script:
```bash
python feature_extraction.py
```

## Output

- **Console** — prints keypoint count, descriptor shape, blob count, contour count, and HOG descriptor length.
- **Dashboard** — a 3×3 grid of plots showing each extraction result.
- **Saved file** — `feature_extraction_results.png` saved in the script's directory.

## Project Structure

```
.
├── feature_extraction.py   # Main script
└── feature_extraction_results.png  # Output (generated on run)
```
