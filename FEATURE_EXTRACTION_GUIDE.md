# Edge-Based Feature Extraction - Technical Guide

**Pneumonia Detection from Chest X-Rays**

This document provides detailed calculation methods for each of the ~43 edge-based features extracted from chest X-ray images using anatomical landmarks (Right Lung, Left Lung, Heart).

---

## Table of Contents

1. [Preprocessing & Mask Generation](#1-preprocessing--mask-generation)
2. [Edge Detection Methods](#2-edge-detection-methods)
3. [Feature Category 1: Lung Edge Integrity (12 features)](#3-lung-edge-integrity-features)
4. [Feature Category 2: Edge Loss & Structural Disruption (8 features)](#4-edge-loss--structural-disruption-features)
5. [Feature Category 3: Opacity Patch Edge Features (10 features)](#5-opacity-patch-edge-features)
6. [Feature Category 4: Zonal Distribution Features (6 features)](#6-zonal-distribution-features)
7. [Feature Category 5: Silhouette Sign Features (3 features)](#7-silhouette-sign-features)
8. [Feature Category 6: Left-Right Asymmetry Features (4 features)](#8-left-right-asymmetry-features)
9. [Implementation Parameters](#9-implementation-parameters)

---

## 1. Preprocessing & Mask Generation

### 1.1 Load Landmarks

```python
# Parse landmarks CSV
landmarks = {
    'RL': [(x1, y1), (x2, y2), ..., (x44, y44)],  # 44 points
    'LL': [(x1, y1), (x2, y2), ..., (x50, y50)],  # 50 points
    'H': [(x1, y1), (x2, y2), ..., (x26, y26)]    # 26 points
}
```

### 1.2 Create Binary Masks

**Algorithm:** Convex Hull

```python
import cv2
import numpy as np

def create_mask(landmarks, image_shape):
    """
    Create binary mask from landmark points.
    
    Args:
        landmarks: List of (x, y) tuples
        image_shape: (height, width)
    
    Returns:
        mask: Binary mask (0/255)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(landmarks, dtype=np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask
```

### 1.3 Lung Zonal Division

**Vertical split into 3 zones:**

```python
def create_zonal_masks(lung_mask):
    """
    Divide lung into upper, middle, lower zones.
    
    Returns:
        upper_mask, middle_mask, lower_mask
    """
    # Find bounding box
    y_coords = np.where(lung_mask > 0)[0]
    y_min, y_max = y_coords.min(), y_coords.max()
    lung_height = y_max - y_min
    
    # Compute zone boundaries
    upper_boundary = y_min + lung_height // 3
    middle_boundary = y_min + 2 * lung_height // 3
    
    # Create zone masks
    upper_mask = lung_mask.copy()
    upper_mask[upper_boundary:, :] = 0
    
    middle_mask = lung_mask.copy()
    middle_mask[:upper_boundary, :] = 0
    middle_mask[middle_boundary:, :] = 0
    
    lower_mask = lung_mask.copy()
    lower_mask[:middle_boundary, :] = 0
    
    return upper_mask, middle_mask, lower_mask
```

---

## 2. Edge Detection Methods

### 2.1 Canny Edge Detection

```python
def compute_canny_edges(image, low_thresh=30, high_thresh=100):
    """
    Compute Canny edges.
    
    Returns:
        Binary edge map (0/255)
    """
    edges = cv2.Canny(image, low_thresh, high_thresh)
    return edges
```

### 2.2 Sobel Gradient Magnitude

```python
def compute_sobel_gradient(image):
    """
    Compute gradient magnitude using Sobel operator.
    
    Returns:
        Gradient magnitude (float, 0-255 scale)
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return gradient_magnitude
```

### 2.3 Edge Skeletonization

```python
from skimage.morphology import skeletonize

def compute_skeleton_edges(edge_map):
    """
    Thin edges to 1-pixel width using morphological skeletonization.
    
    Args:
        edge_map: Binary edge image
    
    Returns:
        Skeletonized edge map
    """
    skeleton = skeletonize(edge_map > 0)
    return (skeleton * 255).astype(np.uint8)
```

### 2.4 Edge Orientation

```python
def compute_edge_orientations(gradient_x, gradient_y):
    """
    Compute edge orientation angles.
    
    Returns:
        Orientation in radians [-π, π]
    """
    orientations = np.arctan2(gradient_y, gradient_x)
    return orientations
```

---

## 3. Lung Edge Integrity Features

**Clinical Rationale:** Normal lungs show rich vascular markings (edges). Pneumonia causes consolidation → edges disappear.

**Features Extracted:** 6 features × 2 lungs = **12 features total**

### 3.1 Edge Density

**Definition:** Proportion of edge pixels within lung region.

**Calculation:**
```python
edge_density = np.sum(edges_masked > 0) / np.sum(lung_mask > 0)
```

**Clinical Interpretation:**
- Normal: 0.15 - 0.30
- Pneumonia: 0.05 - 0.15 (↓ edges due to consolidation)

---

### 3.2 Mean Edge Strength

**Definition:** Average gradient magnitude at edge locations.

**Calculation:**
```python
edge_locations = edges_masked > 0
mean_edge_strength = np.mean(gradient_magnitude[edge_locations])
```

**Clinical Interpretation:**
- Normal: High (sharp vascular edges)
- Pneumonia: Low (blurred boundaries)

---

### 3.3 Std Edge Strength

**Definition:** Standard deviation of edge gradient magnitudes.

**Calculation:**
```python
std_edge_strength = np.std(gradient_magnitude[edge_locations])
```

**Clinical Interpretation:**
- High variance: Mixed pathology (some sharp, some diffuse)
- Low variance: Uniform consolidation

---

### 3.4 Thin Edge Ratio

**Definition:** Proportion of edges that are thin (vascular-like).

**Calculation:**
```python
skeleton_edges = skeletonize(edges_masked)
thin_edge_pixels = np.sum(skeleton_edges > 0)
total_edge_pixels = np.sum(edges_masked > 0)
thin_edge_ratio = thin_edge_pixels / total_edge_pixels if total_edge_pixels > 0 else 0
```

**Clinical Interpretation:**
- Normal: High (≈0.4-0.6, fine vascular markings)
- Pneumonia: Low (thick, blob-like opacities)

---

### 3.5 Thick Edge Ratio

**Definition:** Proportion of edges that are thick (consolidation boundaries).

**Calculation:**
```python
# Dilate original edges to find thick regions
thick_edges = cv2.dilate(edges_masked, kernel=np.ones((3,3)), iterations=1)
thick_only = thick_edges - edges_masked  # Thick component
thick_edge_ratio = np.sum(thick_only > 0) / np.sum(lung_mask > 0)
```

**Clinical Interpretation:**
- Pneumonia: High (large consolidation patches)

---

### 3.6 Orientation Entropy

**Definition:** Randomness of edge orientations (isotropy measure).

**Calculation:**
```python
from scipy.stats import entropy

def compute_orientation_entropy(orientations, edges_masked):
    """
    Compute entropy of edge orientation histogram.
    """
    edge_orientations = orientations[edges_masked > 0]
    # Bin orientations into 18 bins (20° each)
    hist, _ = np.histogram(edge_orientations, bins=18, range=(-np.pi, np.pi))
    hist = hist / hist.sum()  # Normalize
    orientation_entropy = entropy(hist + 1e-10)  # Add epsilon to avoid log(0)
    return orientation_entropy
```

**Clinical Interpretation:**
- Normal: High entropy (edges in all directions)
- Pneumonia: Low entropy (dominant horizontal/vertical opacity boundaries)

---

## 4. Edge Loss & Structural Disruption Features

**Clinical Rationale:** Pneumonia erases expected lung structures, creating smooth white regions with few edges.

**Features Extracted:** 4 features × 2 lungs = **8 features total**

### 4.1 Missing Edge Ratio

**Definition:** Deficit of edges compared to expected normal lung.

**Calculation:**
```python
# Use reference edge density from normal lungs (empirical: ~0.20)
reference_edge_density = 0.20
expected_edge_count = reference_edge_density * np.sum(lung_mask > 0)
observed_edge_count = np.sum(edges_masked > 0)
missing_edge_ratio = max(0, 1 - observed_edge_count / expected_edge_count)
```

**Clinical Interpretation:**
- 0.0: Normal edge density
- 0.5: 50% edge loss (mild consolidation)
- 0.8+: Severe edge loss (extensive consolidation)

---

### 4.2 Low Gradient Area Ratio

**Definition:** Proportion of lung with very low gradients (smooth white regions).

**Calculation:**
```python
low_gradient_threshold = 10  # Empirical threshold for gradient magnitude
low_gradient_mask = (gradient_magnitude < low_gradient_threshold) & (lung_mask > 0)
low_gradient_area_ratio = np.sum(low_gradient_mask) / np.sum(lung_mask > 0)
```

**Clinical Interpretation:**
- Pneumonia: High (consolidation creates smooth opacities)

---

### 4.3 Edge Gap Count

**Definition:** Number of large edge-free regions (gaps in vascular network).

**Calculation:**
```python
# Invert edges: edge-free regions = 1
edge_free = (edges_masked == 0) & (lung_mask > 0)

# Find connected components
num_labels, labels = cv2.connectedComponents(edge_free.astype(np.uint8))

# Count gaps larger than minimum size
min_gap_size = 100  # pixels
edge_gap_count = 0
for i in range(1, num_labels):
    gap_size = np.sum(labels == i)
    if gap_size > min_gap_size:
        edge_gap_count += 1
```

**Clinical Interpretation:**
- Normal: Few small gaps
- Pneumonia: Multiple large gaps (consolidated regions)

---

### 4.4 Mean Gap Length

**Definition:** Average size of edge-free regions.

**Calculation:**
```python
gap_sizes = []
for i in range(1, num_labels):
    gap_size = np.sum(labels == i)
    if gap_size > min_gap_size:
        gap_sizes.append(gap_size)

mean_gap_length = np.mean(gap_sizes) if len(gap_sizes) > 0 else 0
```

**Clinical Interpretation:**
- Larger gaps indicate more extensive consolidation

---

## 5. Opacity Patch Edge Features

**Clinical Rationale:** Pneumonia appears as patchy opacities with characteristic edge properties.

**Features Extracted:** 5 features × 2 lungs = **10 features total**

### 5.1 Number of Opacity Regions

**Definition:** Count of distinct high-intensity patches.

**Calculation:**
```python
# Threshold to find bright regions
lung_pixels = image[lung_mask > 0]
threshold = np.mean(lung_pixels) + 1.0 * np.std(lung_pixels)
opacity_mask = (image > threshold) & (lung_mask > 0)

# Connected components
num_opacity_regions, _ = cv2.connectedComponents(opacity_mask.astype(np.uint8))
num_opacity_regions -= 1  # Exclude background
```

**Clinical Interpretation:**
- Lobar pneumonia: 1-2 large regions
- Bronchopneumonia: Many small patches (5-20)
- Interstitial: Many tiny patches (20+)

---

### 5.2 Largest Patch Ratio

**Definition:** Area of largest opacity patch relative to lung area.

**Calculation:**
```python
patch_sizes = []
for i in range(1, num_opacity_regions + 1):
    patch_size = np.sum(opacity_labels == i)
    patch_sizes.append(patch_size)

largest_patch_ratio = max(patch_sizes) / np.sum(lung_mask > 0) if patch_sizes else 0
```

**Clinical Interpretation:**
- Lobar: High (0.3-0.7, one dominant consolidation)
- Patchy: Low (0.05-0.15, distributed patches)

---

### 5.3 Mean Patch Edge Strength

**Definition:** Average gradient magnitude at opacity boundaries.

**Calculation:**
```python
# Find boundaries of opacity patches
opacity_edges = cv2.Canny(opacity_mask.astype(np.uint8) * 255, 50, 150)
boundary_gradients = gradient_magnitude[opacity_edges > 0]
mean_patch_edge_strength = np.mean(boundary_gradients) if len(boundary_gradients) > 0 else 0
```

**Clinical Interpretation:**
- Sharp edges: Well-defined consolidation (bacterial)
- Blurry edges: Ill-defined opacities (viral/interstitial)

---

### 5.4 Patch Irregularity

**Definition:** Shape complexity of opacity patches (circularity measure).

**Calculation:**
```python
irregularities = []
for i in range(1, num_opacity_regions + 1):
    patch = (opacity_labels == i).astype(np.uint8)
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contour = contours[0]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            # Circularity = 4π × Area / Perimeter²
            # Perfect circle = 1.0, irregular shape < 1.0
            circularity = 4 * np.pi * area / (perimeter ** 2)
            irregularity = 1 - circularity  # Higher = more irregular
            irregularities.append(irregularity)

patch_irregularity = np.mean(irregularities) if irregularities else 0
```

**Clinical Interpretation:**
- Round patches: Organized lobar consolidation
- Irregular patches: Patchy bronchopneumonia

---

### 5.5 Patch Edge Blur

**Definition:** Gradient fall-off rate across opacity boundaries (sharpness).

**Calculation:**
```python
# Dilate opacity mask to get transition zone
dilated = cv2.dilate(opacity_mask.astype(np.uint8), kernel=np.ones((5,5)), iterations=1)
transition_zone = (dilated - opacity_mask.astype(np.uint8)) & (lung_mask > 0)

# Measure gradient in transition zone
transition_gradients = gradient_magnitude[transition_zone > 0]
patch_edge_blur = np.std(transition_gradients)  # High std = sharp, low std = blurry
```

**Clinical Interpretation:**
- Low blur: Sharp consolidation boundaries
- High blur: Diffuse/ground-glass opacities

---

## 6. Zonal Distribution Features

**Clinical Rationale:** Pneumonia often affects lower lung zones (gravity-dependent).

**Features Extracted:** 3 features × 2 lungs = **6 features total**

### 6.1 Lower Zone Edge Density Ratio

**Definition:** Ratio of edge density in lower zone vs entire lung.

**Calculation:**
```python
lower_edges = edges_masked & (lower_zone_mask > 0)
lower_edge_density = np.sum(lower_edges > 0) / np.sum(lower_zone_mask > 0)

total_edge_density = np.sum(edges_masked > 0) / np.sum(lung_mask > 0)

lower_zone_edge_density_ratio = lower_edge_density / total_edge_density if total_edge_density > 0 else 0
```

**Clinical Interpretation:**
- Normal: ≈1.0 (uniform distribution)
- Pneumonia: <0.8 (lower zone has fewer edges due to consolidation)

---

### 6.2 Lower Zone Opacity Ratio

**Definition:** Proportion of lower zone that is opaque.

**Calculation:**
```python
lower_opacity = opacity_mask & (lower_zone_mask > 0)
lower_zone_opacity_ratio = np.sum(lower_opacity > 0) / np.sum(lower_zone_mask > 0)
```

**Clinical Interpretation:**
- Pneumonia: High (consolidation concentrates in lower zones)

---

### 6.3 Upper vs Lower Edge Drop

**Definition:** Edge density difference between upper and lower zones.

**Calculation:**
```python
upper_edges = edges_masked & (upper_zone_mask > 0)
upper_edge_density = np.sum(upper_edges > 0) / np.sum(upper_zone_mask > 0)

lower_edge_density = np.sum(lower_edges > 0) / np.sum(lower_zone_mask > 0)

upper_vs_lower_edge_drop = upper_edge_density - lower_edge_density
```

**Clinical Interpretation:**
- Positive: Upper zone has more edges (typical pneumonia)
- Negative: Lower zone has more edges (atypical)

---

## 7. Silhouette Sign Features

**Clinical Rationale:** Consolidation adjacent to heart obscures the heart border (silhouette sign).

**Features Extracted:** 3 features (global)

### 7.1 Heart-Lung Edge Density

**Definition:** Edge density at the heart-lung interface.

**Calculation:**
```python
# Create interface zone by dilating heart and intersecting with lungs
heart_dilated = cv2.dilate(heart_mask, kernel=np.ones((10,10)), iterations=1)
interface_zone = heart_dilated & ((RL_mask > 0) | (LL_mask > 0))

# Measure edge density in interface
interface_edges = edges & (interface_zone > 0)
heart_lung_edge_density = np.sum(interface_edges > 0) / np.sum(interface_zone > 0)
```

**Clinical Interpretation:**
- Normal: High (sharp heart border)
- Pneumonia: Low (obscured border = silhouette sign)

---

### 7.2 Border Gradient Mean

**Definition:** Average gradient magnitude at heart-lung border.

**Calculation:**
```python
border_gradients = gradient_magnitude[interface_zone > 0]
border_gradient_mean = np.mean(border_gradients)
```

**Clinical Interpretation:**
- Sharp gradient: Clear border (no adjacent pneumonia)
- Weak gradient: Obscured border (adjacent consolidation)

---

### 7.3 Border Continuity Score

**Definition:** Measure of edge continuity along heart border (broken vs continuous).

**Calculation:**
```python
# Find heart contour
heart_contours, _ = cv2.findContours(heart_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if heart_contours:
    contour = heart_contours[0]
    contour_length = cv2.arcLength(contour, True)
    
    # Sample points along contour and check edge presence
    num_samples = 100
    edge_hits = 0
    for i in range(num_samples):
        t = i / num_samples
        point = contour[int(t * len(contour))]
        x, y = point[0]
        if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
            if edges[y, x] > 0:
                edge_hits += 1
    
    border_continuity_score = edge_hits / num_samples
```

**Clinical Interpretation:**
- Score ≈ 1.0: Continuous border (normal)
- Score < 0.5: Broken border (silhouette sign)

---

## 8. Left-Right Asymmetry Features

**Clinical Rationale:** Pneumonia is often unilateral (affects one lung more than the other).

**Features Extracted:** 4 features (global comparisons)

### 8.1 Edge Density Difference

**Definition:** Absolute difference in edge density between lungs.

**Calculation:**
```python
RL_edge_density = np.sum(RL_edges > 0) / np.sum(RL_mask > 0)
LL_edge_density = np.sum(LL_edges > 0) / np.sum(LL_mask > 0)

edge_density_diff = abs(RL_edge_density - LL_edge_density)
```

**Clinical Interpretation:**
- Large difference: Unilateral pneumonia
- Small difference: Bilateral or no pneumonia

---

### 8.2 Mean Gradient Difference

**Definition:** Difference in average gradient strength between lungs.

**Calculation:**
```python
RL_gradients = gradient_magnitude[RL_mask > 0]
LL_gradients = gradient_magnitude[LL_mask > 0]

mean_gradient_diff = abs(np.mean(RL_gradients) - np.mean(LL_gradients))
```

---

### 8.3 Opacity Area Difference

**Definition:** Difference in total opacity area between lungs.

**Calculation:**
```python
RL_opacity = opacity_mask & (RL_mask > 0)
LL_opacity = opacity_mask & (LL_mask > 0)

RL_opacity_ratio = np.sum(RL_opacity > 0) / np.sum(RL_mask > 0)
LL_opacity_ratio = np.sum(LL_opacity > 0) / np.sum(LL_mask > 0)

opacity_area_diff = abs(RL_opacity_ratio - LL_opacity_ratio)
```

---

### 8.4 Missing Edge Difference

**Definition:** Difference in edge loss between lungs.

**Calculation:**
```python
RL_missing_edge_ratio = compute_missing_edge_ratio(RL_edges, RL_mask)
LL_missing_edge_ratio = compute_missing_edge_ratio(LL_edges, LL_mask)

missing_edge_diff = abs(RL_missing_edge_ratio - LL_missing_edge_ratio)
```

---

## 9. Implementation Parameters

### Recommended Thresholds & Constants

```python
# Edge Detection
CANNY_LOW_THRESHOLD = 30
CANNY_HIGH_THRESHOLD = 100
SOBEL_KERNEL_SIZE = 3

# Opacity Detection
OPACITY_STD_MULTIPLIER = 1.0  # threshold = mean + 1.0*std

# Gap Detection
MIN_GAP_SIZE = 100  # pixels

# Reference Values (from normal lungs)
REFERENCE_EDGE_DENSITY = 0.20
LOW_GRADIENT_THRESHOLD = 10

# Interface Zone
HEART_DILATION_KERNEL_SIZE = 10

# Orientation Binning
ORIENTATION_BINS = 18  # 20° per bin
```

### Library Requirements

```python
import cv2
import numpy as np
from scipy.stats import entropy
from skimage.morphology import skeletonize
```

---

## Summary

**Total Features: 43**

| Category                     | Features per Lung | Total Features |
| ---------------------------- | ----------------- | -------------- |
| Lung Edge Integrity          | 6                 | 12             |
| Edge Loss & Disruption       | 4                 | 8              |
| Opacity Patch Edges          | 5                 | 10             |
| Zonal Distribution           | 3                 | 6              |
| Silhouette Sign              | -                 | 3              |
| Left-Right Asymmetry         | -                 | 4              |
| **TOTAL**                    |                   | **43**         |

All features are **numeric** and **Random Forest compatible**.

---

**End of Documentation**
