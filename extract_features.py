"""
Edge-Based Feature Extraction for Pneumonia Detection

This script extracts 43 radiologist-inspired, edge-based features from chest X-rays
using anatomical landmarks (Right Lung, Left Lung, Heart).

Feature Categories:
1. Lung Edge Integrity (12 features)
2. Edge Loss & Structural Disruption (8 features)
3. Opacity Patch Edge Features (10 features)
4. Zonal Distribution Features (6 features)
5. Silhouette Sign Features (3 features)
6. Left-Right Asymmetry Features (4 features)

Usage:
    python extract_features.py
    python extract_features.py --preprocessing-dir preprocessed_data --output features.csv
    python extract_features.py --max-images 20  # Test on subset
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import cv2
import numpy as np
from scipy.stats import entropy
from skimage.morphology import skeletonize


# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

class FeatureConfig:
    """Configuration parameters for feature extraction."""
    
    # Edge Detection
    CANNY_LOW_THRESHOLD = 30
    CANNY_HIGH_THRESHOLD = 100
    SOBEL_KERNEL_SIZE = 3
    
    # Opacity Detection
    OPACITY_STD_MULTIPLIER = 1.0
    
    # Gap Detection
    MIN_GAP_SIZE = 100
    
    # Reference Values
    REFERENCE_EDGE_DENSITY = 0.20
    LOW_GRADIENT_THRESHOLD = 10
    
    # Interface Zone
    HEART_DILATION_KERNEL_SIZE = 10
    
    # Orientation Binning
    ORIENTATION_BINS = 18


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_landmarks_from_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load landmarks from CSV file.
    
    Args:
        csv_path: Path to landmarks CSV
    
    Returns:
        Dictionary with keys 'RL', 'LL', 'H' containing landmark arrays
    """
    landmarks = {'RL': [], 'LL': [], 'H': []}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row['class']
            x, y = int(row['x']), int(row['y'])
            landmarks[class_name].append((x, y))
    
    return {k: np.array(v) for k, v in landmarks.items()}


def create_mask(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create binary mask from landmark points using convex hull.
    
    Args:
        landmarks: Array of (x, y) points
        image_shape: (height, width)
    
    Returns:
        Binary mask (0/255)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    if len(landmarks) == 0:
        return mask
    
    points = landmarks.astype(np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def create_zonal_masks(lung_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide lung mask into upper, middle, lower zones (vertical split).
    
    Returns:
        (upper_mask, middle_mask, lower_mask)
    """
    y_coords = np.where(lung_mask > 0)[0]
    if len(y_coords) == 0:
        return lung_mask.copy(), lung_mask.copy(), lung_mask.copy()
    
    y_min, y_max = y_coords.min(), y_coords.max()
    lung_height = y_max - y_min
    
    upper_boundary = y_min + lung_height // 3
    middle_boundary = y_min + 2 * lung_height // 3
    
    upper_mask = lung_mask.copy()
    upper_mask[upper_boundary:, :] = 0
    
    middle_mask = lung_mask.copy()
    middle_mask[:upper_boundary, :] = 0
    middle_mask[middle_boundary:, :] = 0
    
    lower_mask = lung_mask.copy()
    lower_mask[:middle_boundary, :] = 0
    
    return upper_mask, middle_mask, lower_mask


# ==============================================================================
# EDGE DETECTION
# ==============================================================================

def compute_canny_edges(image: np.ndarray) -> np.ndarray:
    """Compute Canny edge map."""
    return cv2.Canny(
        image,
        FeatureConfig.CANNY_LOW_THRESHOLD,
        FeatureConfig.CANNY_HIGH_THRESHOLD
    )


def compute_sobel_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Sobel gradients.
    
    Returns:
        (gradient_magnitude, gradient_x, gradient_y)
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=FeatureConfig.SOBEL_KERNEL_SIZE)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=FeatureConfig.SOBEL_KERNEL_SIZE)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return gradient_magnitude, sobel_x, sobel_y


def compute_edge_orientations(gradient_x: np.ndarray, gradient_y: np.ndarray) -> np.ndarray:
    """Compute edge orientation angles in radians."""
    return np.arctan2(gradient_y, gradient_x)


# ==============================================================================
# FEATURE EXTRACTION - CATEGORY 1: LUNG EDGE INTEGRITY
# ==============================================================================

def extract_edge_integrity_features(
    image: np.ndarray,
    edges: np.ndarray,
    gradient_magnitude: np.ndarray,
    orientations: np.ndarray,
    lung_mask: np.ndarray,
    lung_prefix: str
) -> Dict[str, float]:
    """
    Extract edge integrity features for one lung.
    
    Returns 6 features.
    """
    features = {}
    
    edges_masked = edges & (lung_mask > 0)
    lung_area = np.sum(lung_mask > 0)
    
    if lung_area == 0:
        return {f"{lung_prefix}_{k}": 0.0 for k in [
            'edge_density', 'mean_edge_strength', 'std_edge_strength',
            'thin_edge_ratio', 'thick_edge_ratio', 'orientation_entropy'
        ]}
    
    # 1. Edge Density
    edge_pixels = np.sum(edges_masked > 0)
    features[f"{lung_prefix}_edge_density"] = edge_pixels / lung_area
    
    # 2 & 3. Mean and Std Edge Strength
    edge_locations = edges_masked > 0
    if np.any(edge_locations):
        edge_gradients = gradient_magnitude[edge_locations]
        features[f"{lung_prefix}_mean_edge_strength"] = float(np.mean(edge_gradients))
        features[f"{lung_prefix}_std_edge_strength"] = float(np.std(edge_gradients))
    else:
        features[f"{lung_prefix}_mean_edge_strength"] = 0.0
        features[f"{lung_prefix}_std_edge_strength"] = 0.0
    
    # 4. Thin Edge Ratio
    if edge_pixels > 0:
        skeleton = skeletonize(edges_masked > 0)
        thin_edge_pixels = np.sum(skeleton)
        features[f"{lung_prefix}_thin_edge_ratio"] = thin_edge_pixels / edge_pixels
    else:
        features[f"{lung_prefix}_thin_edge_ratio"] = 0.0
    
    # 5. Thick Edge Ratio
    thick_edges = cv2.dilate(edges_masked, kernel=np.ones((3, 3)), iterations=1)
    thick_only = thick_edges.astype(int) - edges_masked.astype(int)
    thick_only = np.clip(thick_only, 0, 255).astype(np.uint8)
    features[f"{lung_prefix}_thick_edge_ratio"] = np.sum(thick_only > 0) / lung_area
    
    # 6. Orientation Entropy
    if np.any(edge_locations):
        edge_orientations = orientations[edge_locations]
        hist, _ = np.histogram(edge_orientations, bins=FeatureConfig.ORIENTATION_BINS, 
                              range=(-np.pi, np.pi))
        hist = hist / (hist.sum() + 1e-10)
        features[f"{lung_prefix}_orientation_entropy"] = float(entropy(hist + 1e-10))
    else:
        features[f"{lung_prefix}_orientation_entropy"] = 0.0
    
    return features


# ==============================================================================
# FEATURE EXTRACTION - CATEGORY 2: EDGE LOSS & STRUCTURAL DISRUPTION
# ==============================================================================

def extract_edge_loss_features(
    image: np.ndarray,
    edges: np.ndarray,
    gradient_magnitude: np.ndarray,
    lung_mask: np.ndarray,
    lung_prefix: str
) -> Dict[str, float]:
    """
    Extract edge loss and structural disruption features for one lung.
    
    Returns 4 features.
    """
    features = {}
    
    edges_masked = edges & (lung_mask > 0)
    lung_area = np.sum(lung_mask > 0)
    
    if lung_area == 0:
        return {f"{lung_prefix}_{k}": 0.0 for k in [
            'missing_edge_ratio', 'low_gradient_area_ratio',
            'edge_gap_count', 'mean_gap_length'
        ]}
    
    # 1. Missing Edge Ratio
    expected_edge_count = FeatureConfig.REFERENCE_EDGE_DENSITY * lung_area
    observed_edge_count = np.sum(edges_masked > 0)
    features[f"{lung_prefix}_missing_edge_ratio"] = max(
        0.0, 1.0 - observed_edge_count / (expected_edge_count + 1e-10)
    )
    
    # 2. Low Gradient Area Ratio
    low_gradient_mask = (gradient_magnitude < FeatureConfig.LOW_GRADIENT_THRESHOLD) & (lung_mask > 0)
    features[f"{lung_prefix}_low_gradient_area_ratio"] = np.sum(low_gradient_mask) / lung_area
    
    # 3 & 4. Edge Gap Count and Mean Gap Length
    edge_free = ((edges_masked == 0) & (lung_mask > 0)).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(edge_free)
    
    gap_sizes = []
    for i in range(1, num_labels):
        gap_size = np.sum(labels == i)
        if gap_size > FeatureConfig.MIN_GAP_SIZE:
            gap_sizes.append(gap_size)
    
    features[f"{lung_prefix}_edge_gap_count"] = len(gap_sizes)
    features[f"{lung_prefix}_mean_gap_length"] = float(np.mean(gap_sizes)) if gap_sizes else 0.0
    
    return features


# ==============================================================================
# FEATURE EXTRACTION - CATEGORY 3: OPACITY PATCH EDGE FEATURES
# ==============================================================================

def extract_opacity_patch_features(
    image: np.ndarray,
    gradient_magnitude: np.ndarray,
    lung_mask: np.ndarray,
    lung_prefix: str
) -> Dict[str, float]:
    """
    Extract opacity patch edge features for one lung.
    
    Returns 5 features.
    """
    features = {}
    
    lung_area = np.sum(lung_mask > 0)
    
    if lung_area == 0:
        return {f"{lung_prefix}_{k}": 0.0 for k in [
            'num_opacity_regions', 'largest_patch_ratio', 'mean_patch_edge_strength',
            'patch_irregularity', 'patch_edge_blur'
        ]}
    
    # Compute opacity threshold
    lung_pixels = image[lung_mask > 0]
    if len(lung_pixels) == 0:
        return {f"{lung_prefix}_{k}": 0.0 for k in [
            'num_opacity_regions', 'largest_patch_ratio', 'mean_patch_edge_strength',
            'patch_irregularity', 'patch_edge_blur'
        ]}
    
    threshold = np.mean(lung_pixels) + FeatureConfig.OPACITY_STD_MULTIPLIER * np.std(lung_pixels)
    opacity_mask = ((image > threshold) & (lung_mask > 0)).astype(np.uint8)
    
    # Connected components
    num_labels, labels = cv2.connectedComponents(opacity_mask)
    num_opacity_regions = num_labels - 1  # Exclude background
    
    features[f"{lung_prefix}_num_opacity_regions"] = num_opacity_regions
    
    if num_opacity_regions == 0:
        features[f"{lung_prefix}_largest_patch_ratio"] = 0.0
        features[f"{lung_prefix}_mean_patch_edge_strength"] = 0.0
        features[f"{lung_prefix}_patch_irregularity"] = 0.0
        features[f"{lung_prefix}_patch_edge_blur"] = 0.0
        return features
    
    # Compute patch sizes
    patch_sizes = []
    for i in range(1, num_labels):
        patch_size = np.sum(labels == i)
        patch_sizes.append(patch_size)
    
    # 2. Largest Patch Ratio
    features[f"{lung_prefix}_largest_patch_ratio"] = max(patch_sizes) / lung_area
    
    # 3. Mean Patch Edge Strength
    opacity_edges = cv2.Canny((opacity_mask * 255).astype(np.uint8), 50, 150)
    boundary_gradients = gradient_magnitude[opacity_edges > 0]
    features[f"{lung_prefix}_mean_patch_edge_strength"] = float(np.mean(boundary_gradients)) if len(boundary_gradients) > 0 else 0.0
    
    # 4. Patch Irregularity
    irregularities = []
    for i in range(1, num_labels):
        patch = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = contours[0]
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0 and area > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                irregularity = 1 - circularity
                irregularities.append(irregularity)
    
    features[f"{lung_prefix}_patch_irregularity"] = float(np.mean(irregularities)) if irregularities else 0.0
    
    # 5. Patch Edge Blur
    dilated = cv2.dilate(opacity_mask, kernel=np.ones((5, 5)), iterations=1)
    transition_zone = ((dilated.astype(int) - opacity_mask.astype(int)) > 0) & (lung_mask > 0)
    transition_gradients = gradient_magnitude[transition_zone]
    features[f"{lung_prefix}_patch_edge_blur"] = float(np.std(transition_gradients)) if len(transition_gradients) > 0 else 0.0
    
    return features


# ==============================================================================
# FEATURE EXTRACTION - CATEGORY 4: ZONAL DISTRIBUTION FEATURES
# ==============================================================================

def extract_zonal_features(
    image: np.ndarray,
    edges: np.ndarray,
    lung_mask: np.ndarray,
    lung_prefix: str
) -> Dict[str, float]:
    """
    Extract zonal distribution features for one lung.
    
    Returns 3 features.
    """
    features = {}
    
    upper_mask, middle_mask, lower_mask = create_zonal_masks(lung_mask)
    
    edges_masked = edges & (lung_mask > 0)
    
    # Compute edge densities
    total_edge_density = np.sum(edges_masked > 0) / (np.sum(lung_mask > 0) + 1e-10)
    
    lower_area = np.sum(lower_mask > 0)
    upper_area = np.sum(upper_mask > 0)
    
    if lower_area > 0:
        lower_edges = edges & (lower_mask > 0)
        lower_edge_density = np.sum(lower_edges > 0) / lower_area
        
        # 1. Lower Zone Edge Density Ratio
        features[f"{lung_prefix}_lower_zone_edge_density_ratio"] = lower_edge_density / (total_edge_density + 1e-10)
        
        # 2. Lower Zone Opacity Ratio
        lung_pixels = image[lung_mask > 0]
        if len(lung_pixels) > 0:
            threshold = np.mean(lung_pixels) + FeatureConfig.OPACITY_STD_MULTIPLIER * np.std(lung_pixels)
            opacity_mask = (image > threshold) & (lung_mask > 0)
            lower_opacity = opacity_mask & (lower_mask > 0)
            features[f"{lung_prefix}_lower_zone_opacity_ratio"] = np.sum(lower_opacity) / lower_area
        else:
            features[f"{lung_prefix}_lower_zone_opacity_ratio"] = 0.0
    else:
        features[f"{lung_prefix}_lower_zone_edge_density_ratio"] = 0.0
        features[f"{lung_prefix}_lower_zone_opacity_ratio"] = 0.0
        lower_edge_density = 0.0
    
    # 3. Upper vs Lower Edge Drop
    if upper_area > 0:
        upper_edges = edges & (upper_mask > 0)
        upper_edge_density = np.sum(upper_edges > 0) / upper_area
        features[f"{lung_prefix}_upper_vs_lower_edge_drop"] = upper_edge_density - lower_edge_density
    else:
        features[f"{lung_prefix}_upper_vs_lower_edge_drop"] = 0.0
    
    return features


# ==============================================================================
# FEATURE EXTRACTION - CATEGORY 5: SILHOUETTE SIGN FEATURES
# ==============================================================================

def extract_silhouette_features(
    image: np.ndarray,
    edges: np.ndarray,
    gradient_magnitude: np.ndarray,
    heart_mask: np.ndarray,
    RL_mask: np.ndarray,
    LL_mask: np.ndarray
) -> Dict[str, float]:
    """
    Extract silhouette sign features (heart-lung interface).
    
    Returns 3 features.
    """
    features = {}
    
    # Create interface zone
    heart_dilated = cv2.dilate(
        heart_mask,
        kernel=np.ones((FeatureConfig.HEART_DILATION_KERNEL_SIZE, 
                       FeatureConfig.HEART_DILATION_KERNEL_SIZE)),
        iterations=1
    )
    lung_mask_combined = ((RL_mask > 0) | (LL_mask > 0)).astype(np.uint8)
    interface_zone = (heart_dilated > 0) & (lung_mask_combined > 0)
    
    interface_area = np.sum(interface_zone)
    
    if interface_area == 0:
        return {
            'heart_lung_edge_density': 0.0,
            'border_gradient_mean': 0.0,
            'border_continuity_score': 0.0
        }
    
    # 1. Heart-Lung Edge Density
    interface_edges = edges & interface_zone
    features['heart_lung_edge_density'] = np.sum(interface_edges > 0) / interface_area
    
    # 2. Border Gradient Mean
    border_gradients = gradient_magnitude[interface_zone]
    features['border_gradient_mean'] = float(np.mean(border_gradients))
    
    # 3. Border Continuity Score
    heart_contours, _ = cv2.findContours(heart_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if heart_contours:
        contour = heart_contours[0]
        num_samples = 100
        edge_hits = 0
        
        for i in range(num_samples):
            t = i / num_samples
            idx = int(t * len(contour))
            point = contour[idx][0]
            x, y = point[0], point[1]
            
            if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
                if edges[y, x] > 0:
                    edge_hits += 1
        
        features['border_continuity_score'] = edge_hits / num_samples
    else:
        features['border_continuity_score'] = 0.0
    
    return features


# ==============================================================================
# FEATURE EXTRACTION - CATEGORY 6: LEFT-RIGHT ASYMMETRY FEATURES
# ==============================================================================

def extract_asymmetry_features(
    image: np.ndarray,
    edges: np.ndarray,
    gradient_magnitude: np.ndarray,
    RL_mask: np.ndarray,
    LL_mask: np.ndarray
) -> Dict[str, float]:
    """
    Extract left-right asymmetry features.
    
    Returns 4 features.
    """
    features = {}
    
    RL_area = np.sum(RL_mask > 0)
    LL_area = np.sum(LL_mask > 0)
    
    if RL_area == 0 or LL_area == 0:
        return {
            'edge_density_diff': 0.0,
            'mean_gradient_diff': 0.0,
            'opacity_area_diff': 0.0,
            'missing_edge_diff': 0.0
        }
    
    # 1. Edge Density Difference
    RL_edges = edges & (RL_mask > 0)
    LL_edges = edges & (LL_mask > 0)
    
    RL_edge_density = np.sum(RL_edges > 0) / RL_area
    LL_edge_density = np.sum(LL_edges > 0) / LL_area
    
    features['edge_density_diff'] = abs(RL_edge_density - LL_edge_density)
    
    # 2. Mean Gradient Difference
    RL_gradients = gradient_magnitude[RL_mask > 0]
    LL_gradients = gradient_magnitude[LL_mask > 0]
    
    features['mean_gradient_diff'] = abs(np.mean(RL_gradients) - np.mean(LL_gradients))
    
    # 3. Opacity Area Difference
    RL_pixels = image[RL_mask > 0]
    LL_pixels = image[LL_mask > 0]
    
    if len(RL_pixels) > 0 and len(LL_pixels) > 0:
        RL_threshold = np.mean(RL_pixels) + FeatureConfig.OPACITY_STD_MULTIPLIER * np.std(RL_pixels)
        LL_threshold = np.mean(LL_pixels) + FeatureConfig.OPACITY_STD_MULTIPLIER * np.std(LL_pixels)
        
        RL_opacity = ((image > RL_threshold) & (RL_mask > 0))
        LL_opacity = ((image > LL_threshold) & (LL_mask > 0))
        
        RL_opacity_ratio = np.sum(RL_opacity) / RL_area
        LL_opacity_ratio = np.sum(LL_opacity) / LL_area
        
        features['opacity_area_diff'] = abs(RL_opacity_ratio - LL_opacity_ratio)
    else:
        features['opacity_area_diff'] = 0.0
    
    # 4. Missing Edge Difference
    RL_expected = FeatureConfig.REFERENCE_EDGE_DENSITY * RL_area
    LL_expected = FeatureConfig.REFERENCE_EDGE_DENSITY * LL_area
    
    RL_missing = max(0.0, 1.0 - np.sum(RL_edges > 0) / (RL_expected + 1e-10))
    LL_missing = max(0.0, 1.0 - np.sum(LL_edges > 0) / (LL_expected + 1e-10))
    
    features['missing_edge_diff'] = abs(RL_missing - LL_missing)
    
    return features


# ==============================================================================
# MAIN FEATURE EXTRACTOR CLASS
# ==============================================================================

class PneumoniaFeatureExtractor:
    """Extract edge-based features for pneumonia detection."""
    
    def __init__(self, preprocessing_dir: str = "preprocessed_data"):
        """
        Initialize feature extractor.
        
        Args:
            preprocessing_dir: Directory containing preprocessed images and landmarks
        """
        self.preprocessing_dir = Path(preprocessing_dir)
        self.images_dir = self.preprocessing_dir / "images"
        self.landmarks_dir = self.preprocessing_dir / "landmarks"
        self.summary_csv = self.preprocessing_dir / "preprocessing_summary.csv"
        
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_images': []
        }
    
    def extract_all_features(
        self,
        image_path: str,
        landmarks_path: str
    ) -> Dict[str, float]:
        """
        Extract all 43 features from a single image.
        
        Args:
            image_path: Path to preprocessed image
            landmarks_path: Path to landmarks CSV
        
        Returns:
            Dictionary with all features
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Load landmarks
        landmarks = load_landmarks_from_csv(landmarks_path)
        
        # Create masks
        RL_mask = create_mask(landmarks['RL'], image.shape)
        LL_mask = create_mask(landmarks['LL'], image.shape)
        H_mask = create_mask(landmarks['H'], image.shape)
        
        # Compute edge maps and gradients
        edges = compute_canny_edges(image)
        gradient_magnitude, gradient_x, gradient_y = compute_sobel_gradient(image)
        orientations = compute_edge_orientations(gradient_x, gradient_y)
        
        # Extract features from all categories
        features = {}
        
        # Category 1: Lung Edge Integrity (12 features)
        features.update(extract_edge_integrity_features(
            image, edges, gradient_magnitude, orientations, RL_mask, 'RL'
        ))
        features.update(extract_edge_integrity_features(
            image, edges, gradient_magnitude, orientations, LL_mask, 'LL'
        ))
        
        # Category 2: Edge Loss & Structural Disruption (8 features)
        features.update(extract_edge_loss_features(
            image, edges, gradient_magnitude, RL_mask, 'RL'
        ))
        features.update(extract_edge_loss_features(
            image, edges, gradient_magnitude, LL_mask, 'LL'
        ))
        
        # Category 3: Opacity Patch Edge Features (10 features)
        features.update(extract_opacity_patch_features(
            image, gradient_magnitude, RL_mask, 'RL'
        ))
        features.update(extract_opacity_patch_features(
            image, gradient_magnitude, LL_mask, 'LL'
        ))
        
        # Category 4: Zonal Distribution Features (6 features)
        features.update(extract_zonal_features(
            image, edges, RL_mask, 'RL'
        ))
        features.update(extract_zonal_features(
            image, edges, LL_mask, 'LL'
        ))
        
        # Category 5: Silhouette Sign Features (3 features)
        features.update(extract_silhouette_features(
            image, edges, gradient_magnitude, H_mask, RL_mask, LL_mask
        ))
        
        # Category 6: Left-Right Asymmetry Features (4 features)
        features.update(extract_asymmetry_features(
            image, edges, gradient_magnitude, RL_mask, LL_mask
        ))
        
        return features
    
    def process_dataset(self, max_images: Optional[int] = None) -> List[Dict[str, any]]:
        """
        Extract features from all images in the dataset.
        
        Args:
            max_images: Maximum number of images to process
        
        Returns:
            List of feature dictionaries
        """
        print("=" * 70)
        print("EDGE-BASED FEATURE EXTRACTION")
        print("=" * 70)
        print(f"Preprocessing directory: {self.preprocessing_dir}")
        print(f"Total features per image: 43")
        print("=" * 70 + "\n")
        
        # Load preprocessing summary
        if not self.summary_csv.exists():
            raise FileNotFoundError(f"Preprocessing summary not found: {self.summary_csv}")
        
        dataset_records = []
        with open(self.summary_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset_records.append(row)
        
        if max_images is not None:
            dataset_records = dataset_records[:max_images]
        
        self.stats['total'] = len(dataset_records)
        print(f"Processing {len(dataset_records)} images\n")
        
        # Process each image
        results = []
        for idx, record in enumerate(dataset_records, 1):
            image_name = record['image_name']
            label = int(record['label'])
            split = record['split']
            success = record.get('success', 'True').lower() == 'true'
            
            if not success:
                print(f"[{idx}/{len(dataset_records)}] Skipping {image_name} (preprocessing failed)")
                continue
            
            # Construct paths
            preprocessed_filename = Path(image_name).stem + '_preprocessed.png'
            landmarks_filename = Path(image_name).stem + '_landmarks.csv'
            
            image_path = self.images_dir / preprocessed_filename
            landmarks_path = self.landmarks_dir / landmarks_filename
            
            print(f"[{idx}/{len(dataset_records)}] Extracting features: {image_name}")
            
            try:
                features = self.extract_all_features(str(image_path), str(landmarks_path))
                
                # Add metadata
                result = {
                    'image_name': image_name,
                    'label': label,
                    'split': split,
                    **features
                }
                
                results.append(result)
                self.stats['success'] += 1
                print(f"  ✓ Success ({len(features)} features extracted)")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                self.stats['failed'] += 1
                self.stats['failed_images'].append({
                    'image_name': image_name,
                    'error': str(e)
                })
        
        return results
    
    def save_features(
        self,
        features_list: List[Dict[str, any]],
        output_csv: str = "features.csv",
        output_json: str = "features.json"
    ) -> None:
        """
        Save extracted features to CSV and JSON files.
        
        Args:
            features_list: List of feature dictionaries
            output_csv: Output CSV path
            output_json: Output JSON path
        """
        print("\n" + "=" * 70)
        print("SAVING FEATURES")
        print("=" * 70)
        
        if not features_list:
            print("⚠ Warning: No features to save")
            return
        
        # Save CSV
        fieldnames = list(features_list[0].keys())
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features_list)
        
        print(f"✓ Features CSV saved: {output_csv}")
        print(f"  - {len(features_list)} samples")
        print(f"  - {len(fieldnames) - 3} features (+ metadata)")
        
        # Save JSON
        feature_names = [k for k in fieldnames if k not in ['image_name', 'label', 'split']]
        
        json_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_images': len(features_list),
                'num_features': len(feature_names),
                'feature_names': feature_names
            },
            'features': features_list
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✓ Features JSON saved: {output_json}")
        
        # Print summary
        print("\n" + "-" * 70)
        print("Feature Extraction Summary:")
        print("-" * 70)
        print(f"Total images processed: {self.stats['total']}")
        print(f"Successfully extracted: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        
        if self.stats['failed'] > 0:
            print(f"\nFailed images:")
            for failed in self.stats['failed_images']:
                print(f"  - {failed['image_name']}: {failed['error']}")
        
        success_rate = (self.stats['success'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")
        print("=" * 70)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Extract edge-based features for pneumonia detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--preprocessing-dir",
        type=str,
        default="preprocessed_data",
        help="Directory containing preprocessed images and landmarks"
    )
    
    parser.add_argument(
        "--output-csv",
        type=str,
        default="features.csv",
        help="Output CSV filename for features"
    )
    
    parser.add_argument(
        "--output-json",
        type=str,
        default="features.json",
        help="Output JSON filename for features"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = PneumoniaFeatureExtractor(preprocessing_dir=args.preprocessing_dir)
    
    # Extract features
    features_list = extractor.process_dataset(max_images=args.max_images)
    
    # Save features
    extractor.save_features(features_list, args.output_csv, args.output_json)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"Features saved to: {args.output_csv}")
    print(f"\nNext steps:")
    print(f"1. Review features.csv to verify extracted features")
    print(f"2. Train Random Forest model: python train_models.py")


if __name__ == "__main__":
    main()
