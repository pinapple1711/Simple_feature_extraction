import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# LOAD IMAGE
# ==========================================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_rgb, gray


# ==========================================
# 1. EDGE DETECTION (Canny)
# ==========================================
def extract_edges(gray):
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return edges


# ==========================================
# 2. CORNER DETECTION (Harris)
# ==========================================
def extract_corners(gray, img_rgb):
    gray_float = np.float32(gray)
    corners = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)  # Dilate to make corners visible

    img_corners = img_rgb.copy()
    img_corners[corners > 0.01 * corners.max()] = [255, 0, 0]  # Mark in red
    return img_corners


# ==========================================
# 3. KEYPOINTS & DESCRIPTORS (ORB)
# ==========================================
def extract_orb_keypoints(gray, img_rgb):
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    img_kp = cv2.drawKeypoints(
        img_rgb, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    print(f"  ORB Keypoints detected: {len(keypoints)}")
    print(f"  Descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")
    return img_kp, keypoints, descriptors


# ==========================================
# 4. BLOB DETECTION (SimpleBlobDetector)
# ==========================================
def extract_blobs(gray, img_rgb):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = True
    params.minCircularity = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    blobs = detector.detect(gray)

    img_blobs = cv2.drawKeypoints(
        img_rgb, blobs, None,
        color=(255, 0, 255),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    print(f"  Blobs detected: {len(blobs)}")
    return img_blobs


# ==========================================
# 5. CONTOUR DETECTION
# ==========================================
def extract_contours(gray, img_rgb):
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = img_rgb.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 200, 255), 2)
    print(f"  Contours detected: {len(contours)}")
    return img_contours


# ==========================================
# 6. HOG — HISTOGRAM OF ORIENTED GRADIENTS
# ==========================================
def extract_hog(gray):
    win_size = (64, 64)
    resized = cv2.resize(gray, win_size)

    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    descriptor = hog.compute(resized)
    print(f"  HOG descriptor length: {descriptor.shape[0]}")
    return descriptor


# ==========================================
# 7. COLOR HISTOGRAM
# ==========================================
def extract_color_histogram(img_rgb):
    colors = ('r', 'g', 'b')
    histograms = {}
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        histograms[color] = hist
    return histograms


# ==========================================
# VISUALIZE ALL RESULTS
# ==========================================
def visualize_results(img_rgb, gray, edges, img_corners, img_kp, img_blobs, img_contours, histograms):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("OpenCV Feature Extraction", fontsize=18, fontweight='bold')

    panels = [
        (img_rgb,       "Original Image"),
        (gray,          "Grayscale",        "gray"),
        (edges,         "Edges (Canny)",    "gray"),
        (img_corners,   "Corners (Harris)"),
        (img_kp,        "Keypoints (ORB)"),
        (img_blobs,     "Blobs"),
        (img_contours,  "Contours"),
    ]

    for idx, panel in enumerate(panels):
        ax = axes[idx // 3][idx % 3]
        cmap = panel[2] if len(panel) == 3 else None
        ax.imshow(panel[0], cmap=cmap)
        ax.set_title(panel[1], fontsize=11)
        ax.axis('off')

    # Color Histogram in last panel
    ax_hist = axes[2][1]
    for color, hist in histograms.items():
        ax_hist.plot(hist, color=color, alpha=0.7)
    ax_hist.set_title("Color Histogram")
    ax_hist.set_xlim([0, 256])
    ax_hist.set_xlabel("Pixel Intensity")
    ax_hist.set_ylabel("Frequency")

    axes[2][2].axis('off')  # Hide unused panel

    plt.tight_layout()
    plt.savefig("feature_extraction_results.png", dpi=150, bbox_inches='tight')
    plt.show()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\ThinkPad\Pictures\shopping.webp"  # <-- Change this to your image path

    print("Loading image...")
    img_rgb, gray = load_image(IMAGE_PATH)

    print("\n[1] Extracting edges...")
    edges = extract_edges(gray)

    print("\n[2] Detecting corners...")
    img_corners = extract_corners(gray, img_rgb)

    print("\n[3] Extracting ORB keypoints & descriptors...")
    img_kp, keypoints, descriptors = extract_orb_keypoints(gray, img_rgb)

    print("\n[4] Detecting blobs...")
    img_blobs = extract_blobs(gray, img_rgb)

    print("\n[5] Detecting contours...")
    img_contours = extract_contours(gray, img_rgb)

    print("\n[6] Computing HOG descriptor...")
    hog_descriptor = extract_hog(gray)

    print("\n[7] Computing color histogram...")
    histograms = extract_color_histogram(img_rgb)

    print("\nVisualizing results...")
    visualize_results(img_rgb, gray, edges, img_corners, img_kp, img_blobs, img_contours, histograms)