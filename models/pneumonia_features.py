"""
Phase 3: Pneumonia Feature Extraction from Chest X-rays

This script extracts clinical features from chest X-rays that radiologists use
to diagnose pneumonia. It combines segmentation landmarks with image analysis
to detect:

1. Consolidation - Dense white/gray patches (fluid-filled alveoli)
2. Air Bronchograms - Black branching lines in white regions
3. Texture Loss - Blurred or erased normal lung markings
4. Silhouette Sign - Loss of heart/diaphragm borders
5. Pleural Effusion - Fluid at lung base
6. Pattern Analysis - Lobar vs patchy distribution

Features are extracted using:
- Landmark-based ROI extraction (from segmentation_landmarks.py)
- Intensity analysis within lung fields
- Texture descriptors (LBP, Gabor filters)
- Edge detection for borders
- Statistical measures
"""

import os
import sys
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Fix Windows encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import landmark extractor
from .hybridgnet_landmarks import ChestXrayLandmarkCentralizer


class PneumoniaFeatureExtractor:
    """
    Extracts pneumonia-specific features from chest X-rays using
    anatomical landmarks and radiological signs.
    """

    def __init__(
        self,
        landmark_centralizer: Optional[ChestXrayLandmarkCentralizer] = None,
        output_dir: str = "pneumonia_features",
    ):
        self.centralizer = landmark_centralizer or ChestXrayLandmarkCentralizer()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # =========================================================================
    # PHASE 2 PREPROCESSING PLAN (Documented but not implemented here)
    # =========================================================================
    """
    PHASE 2 PREPROCESSING STRATEGY:
    
    Before feature extraction, images should undergo preprocessing:
    
    1. IMAGE STANDARDIZATION:
       - Resize to consistent dimensions (1024x1024 for segmentation)
       - Pad to square maintaining aspect ratio
       - Normalize intensity values (0-1 or z-score normalization)
       - CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast
    
    2. SEGMENTATION & LANDMARKS:
       - Run HybridGNet model via segmentation_landmarks.py
       - Extract 120 landmark points (44 right lung, 50 left lung, 26 heart)
       - Generate dense segmentation masks from landmarks (convex hull or interpolation)
       - Store landmark coordinates for ROI-based analysis
    
    3. EDGE ENHANCEMENT:
       - Apply Canny edge detection for border analysis (silhouette sign)
       - Sobel gradients for texture directionality
       - Morphological operations (dilation/erosion) for region refinement
    
    4. TEXTURE PREPARATION:
       - Calculate Local Binary Patterns (LBP) for texture analysis
       - Generate Gabor filter bank responses (multiple orientations/scales)
       - GLCM (Gray Level Co-occurrence Matrix) for texture statistics
    
    5. AUGMENTATION (for training data):
       - Random rotation (±10 degrees)
       - Brightness/contrast adjustments
       - Elastic deformations (mild)
       - Maintain anatomical plausibility
    
    6. QUALITY CONTROL:
       - Remove low-quality images (excessive noise, artifacts)
       - Verify landmark extraction success
       - Log preprocessing failures
    
    Implementation: Create preprocess_xrays.py script that:
    - Loads images from reorganized dataset
    - Applies preprocessing pipeline
    - Extracts and stores landmarks
    - Saves preprocessed images + metadata (JSON)
    - Generates preprocessing report (CSV with quality metrics)
    """

    # =========================================================================
    # CORE FEATURE EXTRACTION
    # =========================================================================

    def extract_features(
        self,
        image_path: str,
        landmarks: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Extract all pneumonia-related features from a chest X-ray.
        Features are numbered from 1 to approximately 100 for machine learning.

        Args:
            image_path: Path to chest X-ray image
            landmarks: Pre-computed landmarks (120, 2). If None, computed automatically.

        Returns:
            Dictionary containing all extracted features with numeric indexing
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_normalized = img.astype(np.float32) / 255.0

        # Get landmarks if not provided
        if landmarks is None:
            landmarks = self.centralizer.process_image(image_path, save_to_file=False)

        # Split landmarks by organ
        right_lung_landmarks = landmarks[:44]
        left_lung_landmarks = landmarks[44:94]
        heart_landmarks = landmarks[94:120]

        # Extract features with descriptive names
        features = {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
        }

        # 1. Consolidation features
        features.update(self._extract_consolidation_features(
            img_normalized, right_lung_landmarks, left_lung_landmarks
        ))

        # 2. Air bronchogram features
        features.update(self._extract_air_bronchogram_features(
            img_normalized, right_lung_landmarks, left_lung_landmarks
        ))

        # 3. Texture loss features
        features.update(self._extract_texture_loss_features(
            img_normalized, right_lung_landmarks, left_lung_landmarks
        ))

        # 4. Silhouette sign features
        features.update(self._extract_silhouette_sign_features(
            img_normalized, heart_landmarks, right_lung_landmarks, left_lung_landmarks
        ))

        # 5. Pleural effusion features
        features.update(self._extract_pleural_effusion_features(
            img_normalized, right_lung_landmarks, left_lung_landmarks
        ))

        # 6. Pattern distribution features
        features.update(self._extract_pattern_distribution_features(
            img_normalized, right_lung_landmarks, left_lung_landmarks
        ))

        return features
    
    def convert_features_to_numbered_format(
        self, 
        features_list: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[int, str], List[str]]:
        """
        Convert named features to numbered format for machine learning.
        
        Args:
            features_list: List of feature dictionaries with named features
            
        Returns:
            - Feature matrix (n_samples, n_features) with numeric features only
            - Feature mapping dictionary {feature_number: feature_name}
            - List of image names corresponding to each row
        """
        if not features_list:
            return np.array([]), {}, []
        
        # Extract only numeric features (exclude metadata)
        numeric_features = {}
        for feature_dict in features_list:
            for key, value in feature_dict.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    if key not in numeric_features:
                        numeric_features[key] = []
                    numeric_features[key].append(value)
        
        # Sort feature names for consistent ordering
        feature_names = sorted(numeric_features.keys())
        
        # Create feature matrix
        feature_matrix = np.array([numeric_features[name] for name in feature_names]).T
        
        # Create feature number to name mapping (1-indexed for readability)
        feature_mapping = {i+1: name for i, name in enumerate(feature_names)}
        
        # Extract image names
        image_names = [f.get('image_name', f.get('image_path', f'sample_{i}')) 
                      for i, f in enumerate(features_list)]
        
        print(f"\n✓ Converted to numbered format: {feature_matrix.shape[1]} features, "
              f"{feature_matrix.shape[0]} samples")
        
        return feature_matrix, feature_mapping, image_names

    def _create_lung_mask(self, landmarks: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """Create binary mask from landmarks using convex hull."""
        mask = np.zeros(img_shape, dtype=np.uint8)
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    # -------------------------------------------------------------------------
    # 1. CONSOLIDATION DETECTION
    # -------------------------------------------------------------------------

    def _extract_consolidation_features(
        self,
        img: np.ndarray,
        right_lung: np.ndarray,
        left_lung: np.ndarray,
    ) -> Dict[str, float]:
        """
        Detect consolidation: dense white/gray patches in lung fields.
        
        Consolidation = alveoli filled with fluid/pus → high intensity regions
        where air (dark) should be.
        """
        features = {}

        for lung_name, landmarks in [("right", right_lung), ("left", left_lung)]:
            # Create lung mask
            mask = self._create_lung_mask(landmarks, img.shape)
            lung_pixels = img[mask > 0]

            if len(lung_pixels) == 0:
                continue

            # Intensity statistics
            features[f"{lung_name}_lung_mean_intensity"] = float(np.mean(lung_pixels))
            features[f"{lung_name}_lung_std_intensity"] = float(np.std(lung_pixels))
            features[f"{lung_name}_lung_max_intensity"] = float(np.max(lung_pixels))

            # High-intensity region detection (consolidation candidate)
            threshold = np.mean(lung_pixels) + 1.5 * np.std(lung_pixels)
            consolidation_pixels = lung_pixels[lung_pixels > threshold]
            features[f"{lung_name}_lung_consolidation_ratio"] = float(
                len(consolidation_pixels) / len(lung_pixels)
            )

            # Distribution statistics
            features[f"{lung_name}_lung_intensity_skew"] = float(skew(lung_pixels))
            features[f"{lung_name}_lung_intensity_kurtosis"] = float(kurtosis(lung_pixels))

            # Regional opacity (divide lung into zones)
            bbox = cv2.boundingRect(landmarks.astype(np.int32))
            x, y, w, h = bbox
            upper_zone = img[y:y+h//3, x:x+w]
            middle_zone = img[y+h//3:y+2*h//3, x:x+w]
            lower_zone = img[y+2*h//3:y+h, x:x+w]

            features[f"{lung_name}_lung_upper_mean"] = float(np.mean(upper_zone[upper_zone > 0]))
            features[f"{lung_name}_lung_middle_mean"] = float(np.mean(middle_zone[middle_zone > 0]))
            features[f"{lung_name}_lung_lower_mean"] = float(np.mean(lower_zone[lower_zone > 0]))

        return features

    # -------------------------------------------------------------------------
    # 2. AIR BRONCHOGRAM DETECTION
    # -------------------------------------------------------------------------

    def _extract_air_bronchogram_features(
        self,
        img: np.ndarray,
        right_lung: np.ndarray,
        left_lung: np.ndarray,
    ) -> Dict[str, float]:
        """
        Detect air bronchograms: black branching structures in white regions.
        
        Very strong pneumonia indicator - bronchi (air) visible against
        consolidated (white) lung tissue.
        """
        features = {}

        # Apply CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply((img * 255).astype(np.uint8))
        img_clahe = img_clahe.astype(np.float32) / 255.0

        for lung_name, landmarks in [("right", right_lung), ("left", left_lung)]:
            mask = self._create_lung_mask(landmarks, img.shape)

            # Detect dark linear structures (potential bronchi)
            # Use morphological operations
            kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            
            # Black-hat transform to detect dark structures
            blackhat = cv2.morphologyEx(
                (img_clahe * 255).astype(np.uint8),
                cv2.MORPH_BLACKHAT,
                kernel_line,
            )
            blackhat_masked = cv2.bitwise_and(blackhat, blackhat, mask=mask)

            # Count significant dark line structures
            _, thresh = cv2.threshold(blackhat_masked, 10, 255, cv2.THRESH_BINARY)
            features[f"{lung_name}_lung_dark_line_density"] = float(
                np.sum(thresh > 0) / np.sum(mask > 0) if np.sum(mask) > 0 else 0
            )

            # Edge density in high-intensity regions (bronchogram pattern)
            edges = cv2.Canny((img_clahe * 255).astype(np.uint8), 50, 150)
            edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
            
            high_intensity_mask = (img > (np.mean(img[mask > 0]) + np.std(img[mask > 0]))).astype(np.uint8) * 255
            high_intensity_mask = cv2.bitwise_and(high_intensity_mask, high_intensity_mask, mask=mask)
            
            edges_in_bright = cv2.bitwise_and(edges_masked, edges_masked, mask=high_intensity_mask)
            features[f"{lung_name}_lung_bronchogram_score"] = float(
                np.sum(edges_in_bright > 0) / np.sum(high_intensity_mask > 0)
                if np.sum(high_intensity_mask) > 0 else 0
            )

        return features

    # -------------------------------------------------------------------------
    # 3. TEXTURE LOSS DETECTION
    # -------------------------------------------------------------------------

    def _extract_texture_loss_features(
        self,
        img: np.ndarray,
        right_lung: np.ndarray,
        left_lung: np.ndarray,
    ) -> Dict[str, float]:
        """
        Detect loss of normal lung markings (texture).
        
        Normal lungs show fine vascular markings. Pneumonia causes
        blurred or erased texture.
        """
        features = {}

        for lung_name, landmarks in [("right", right_lung), ("left", left_lung)]:
            mask = self._create_lung_mask(landmarks, img.shape)

            # Local Binary Pattern (texture descriptor)
            lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
            lbp_masked = lbp[mask > 0]
            
            if len(lbp_masked) > 0:
                # LBP histogram entropy (lower = less texture)
                hist, _ = np.histogram(lbp_masked, bins=10, density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
                features[f"{lung_name}_lung_lbp_entropy"] = float(entropy)

            # Gray Level Co-occurrence Matrix (GLCM)
            img_uint8 = (img * 255).astype(np.uint8)
            roi = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)
            
            # Calculate GLCM
            glcm = graycomatrix(
                roi,
                distances=[1],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                symmetric=True,
                normed=True,
            )
            
            # GLCM properties
            features[f"{lung_name}_lung_glcm_contrast"] = float(graycoprops(glcm, "contrast").mean())
            features[f"{lung_name}_lung_glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
            features[f"{lung_name}_lung_glcm_energy"] = float(graycoprops(glcm, "energy").mean())
            features[f"{lung_name}_lung_glcm_correlation"] = float(graycoprops(glcm, "correlation").mean())

            # Edge density (normal lungs have rich vascular markings)
            edges = cv2.Canny((img * 255).astype(np.uint8), 30, 100)
            edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
            features[f"{lung_name}_lung_edge_density"] = float(
                np.sum(edges_masked > 0) / np.sum(mask > 0) if np.sum(mask) > 0 else 0
            )

        return features

    # -------------------------------------------------------------------------
    # 4. SILHOUETTE SIGN DETECTION
    # -------------------------------------------------------------------------

    def _extract_silhouette_sign_features(
        self,
        img: np.ndarray,
        heart: np.ndarray,
        right_lung: np.ndarray,
        left_lung: np.ndarray,
    ) -> Dict[str, float]:
        """
        Detect silhouette sign: loss of heart/diaphragm borders.
        
        When lung adjacent to heart becomes opaque (pneumonia),
        the heart border disappears.
        """
        features = {}

        # Create masks
        heart_mask = self._create_lung_mask(heart, img.shape)
        right_mask = self._create_lung_mask(right_lung, img.shape)
        left_mask = self._create_lung_mask(left_lung, img.shape)

        # Detect edges
        edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)

        # Find heart border pixels
        heart_contours, _ = cv2.findContours(heart_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if heart_contours:
            heart_border = np.zeros_like(heart_mask)
            cv2.drawContours(heart_border, heart_contours, -1, 255, thickness=5)

            # Check edge density at heart-lung interface
            # Right heart border
            right_interface = cv2.bitwise_and(heart_border, heart_border, mask=right_mask)
            edges_at_right_interface = cv2.bitwise_and(edges, edges, mask=right_interface)
            features["right_heart_border_edge_density"] = float(
                np.sum(edges_at_right_interface > 0) / np.sum(right_interface > 0)
                if np.sum(right_interface) > 0 else 0
            )

            # Left heart border
            left_interface = cv2.bitwise_and(heart_border, heart_border, mask=left_mask)
            edges_at_left_interface = cv2.bitwise_and(edges, edges, mask=left_interface)
            features["left_heart_border_edge_density"] = float(
                np.sum(edges_at_left_interface > 0) / np.sum(left_interface > 0)
                if np.sum(left_interface) > 0 else 0
            )

            # Intensity difference at border (lower = silhouette sign)
            for side, lung_mask in [("right", right_mask), ("left", left_mask)]:
                interface_region = cv2.bitwise_and(heart_border, heart_border, mask=lung_mask)
                if np.sum(interface_region) > 0:
                    # Dilate to get adjacent regions
                    kernel = np.ones((15, 15), np.uint8)
                    expanded = cv2.dilate(interface_region, kernel, iterations=1)
                    
                    heart_side_pixels = img[cv2.bitwise_and(expanded, expanded, mask=heart_mask) > 0]
                    lung_side_pixels = img[cv2.bitwise_and(expanded, expanded, mask=lung_mask) > 0]
                    
                    if len(heart_side_pixels) > 0 and len(lung_side_pixels) > 0:
                        intensity_diff = abs(np.mean(heart_side_pixels) - np.mean(lung_side_pixels))
                        features[f"{side}_heart_border_intensity_diff"] = float(intensity_diff)

        return features

    # -------------------------------------------------------------------------
    # 5. PLEURAL EFFUSION DETECTION
    # -------------------------------------------------------------------------

    def _extract_pleural_effusion_features(
        self,
        img: np.ndarray,
        right_lung: np.ndarray,
        left_lung: np.ndarray,
    ) -> Dict[str, float]:
        """
        Detect pleural effusion: fluid at lung base.
        
        Appears as blunted costophrenic angle and increased
        opacity at lower lung zones.
        """
        features = {}

        for lung_name, landmarks in [("right", right_lung), ("left", left_lung)]:
            # Find lowest points (lung base)
            lowest_y = np.max(landmarks[:, 1])
            base_region_height = int(0.2 * (np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])))
            
            base_landmarks = landmarks[landmarks[:, 1] > (lowest_y - base_region_height)]
            
            if len(base_landmarks) > 0:
                base_mask = self._create_lung_mask(base_landmarks, img.shape)
                base_pixels = img[base_mask > 0]
                
                if len(base_pixels) > 0:
                    # Base opacity
                    features[f"{lung_name}_lung_base_mean_intensity"] = float(np.mean(base_pixels))
                    features[f"{lung_name}_lung_base_std_intensity"] = float(np.std(base_pixels))
                    
                    # Costophrenic angle sharpness (acute=normal, blunt=effusion)
                    # Approximate by curvature at lowest point
                    base_points = landmarks[landmarks[:, 1] > (lowest_y - base_region_height * 2)]
                    if len(base_points) > 3:
                        # Sort by x to get contour order
                        sorted_points = base_points[np.argsort(base_points[:, 0])]
                        
                        # Calculate local curvature at bottom
                        mid_idx = len(sorted_points) // 2
                        if mid_idx > 1 and mid_idx < len(sorted_points) - 1:
                            p1 = sorted_points[mid_idx - 2]
                            p2 = sorted_points[mid_idx]
                            p3 = sorted_points[mid_idx + 2]
                            
                            # Angle between vectors
                            v1 = p2 - p1
                            v2 = p3 - p2
                            
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                            angle = np.arccos(np.clip(cos_angle, -1, 1))
                            features[f"{lung_name}_lung_base_angle"] = float(np.degrees(angle))

        return features

    # -------------------------------------------------------------------------
    # 6. PATTERN DISTRIBUTION ANALYSIS
    # -------------------------------------------------------------------------

    def _extract_pattern_distribution_features(
        self,
        img: np.ndarray,
        right_lung: np.ndarray,
        left_lung: np.ndarray,
    ) -> Dict[str, float]:
        """
        Analyze distribution patterns: lobar vs patchy vs interstitial.
        
        - Lobar pneumonia: one whole lobe consolidated
        - Bronchopneumonia: multiple small patches
        - Interstitial: fine linear patterns
        """
        features = {}

        for lung_name, landmarks in [("right", right_lung), ("left", left_lung)]:
            mask = self._create_lung_mask(landmarks, img.shape)
            
            # Threshold to find opaque regions
            lung_pixels = img[mask > 0]
            if len(lung_pixels) == 0:
                continue
                
            threshold = np.mean(lung_pixels) + 1.0 * np.std(lung_pixels)
            opacity_mask = ((img > threshold) & (mask > 0)).astype(np.uint8) * 255
            
            # Find connected components (patches)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                opacity_mask, connectivity=8
            )
            
            # Filter small noise
            min_area = 100
            valid_components = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area]
            
            features[f"{lung_name}_lung_num_patches"] = len(valid_components)
            
            if len(valid_components) > 0:
                areas = [stats[i, cv2.CC_STAT_AREA] for i in valid_components]
                features[f"{lung_name}_lung_largest_patch_area"] = float(max(areas))
                features[f"{lung_name}_lung_mean_patch_area"] = float(np.mean(areas))
                features[f"{lung_name}_lung_total_opacity_ratio"] = float(
                    sum(areas) / np.sum(mask > 0)
                )
                
                # Distribution pattern
                # Lobar: few large patches
                # Patchy: many small-medium patches
                # Interstitial: many very small patches
                if len(valid_components) == 1:
                    pattern_type = "lobar"
                elif len(valid_components) > 10 and np.mean(areas) < 500:
                    pattern_type = "interstitial"
                else:
                    pattern_type = "patchy"
                    
                features[f"{lung_name}_lung_pattern_type"] = pattern_type

        return features

    # =========================================================================
    # MACHINE LEARNING MODELS - DECISION TREE & RANDOM FOREST
    # =========================================================================

    def train_decision_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_mapping: Dict[int, str],
        test_size: float = 0.2,
        random_state: int = 42,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
    ) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
        """
        Train a decision tree classifier and extract feature importances.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0=Normal, 1=Pneumonia)
            feature_mapping: Dictionary mapping feature numbers to names
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in leaf node
            
        Returns:
            - Trained DecisionTreeClassifier
            - Dictionary containing model metrics and feature importances
        """
        print(f"\n{'='*70}")
        print(f"TRAINING DECISION TREE CLASSIFIER")
        print(f"{'='*70}\n")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {X.shape[1]}")
        
        # Train decision tree
        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        
        print("\nTraining model...")
        dt_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = dt_model.predict(X_train)
        y_test_pred = dt_model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\n✓ Training Accuracy: {train_accuracy:.4f}")
        print(f"✓ Test Accuracy: {test_accuracy:.4f}")
        
        print("\nClassification Report (Test Set):")
        # Get unique labels present in test set
        unique_labels = sorted(set(y_test) | set(y_test_pred))
        target_names = ['Normal' if l == 0 else 'Pneumonia' for l in unique_labels]
        print(classification_report(y_test, y_test_pred, 
                                   labels=unique_labels,
                                   target_names=target_names,
                                   zero_division=0))
        
        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, y_test_pred, labels=unique_labels)
        print(cm)
        
        # Extract feature importances (weights)
        feature_importances = dt_model.feature_importances_
        
        # Create sorted list of (feature_number, feature_name, importance)
        importance_list = []
        for i, importance in enumerate(feature_importances):
            feature_num = i + 1
            feature_name = feature_mapping.get(feature_num, f"feature_{feature_num}")
            importance_list.append({
                'feature_number': feature_num,
                'feature_name': feature_name,
                'importance': float(importance)
            })
        
        # Sort by importance
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        print(f"\nTop 10 Most Important Features (Decision Tree):")
        for i, item in enumerate(importance_list[:10], 1):
            print(f"  {i}. Feature {item['feature_number']:3d} "
                  f"({item['feature_name'][:50]}): {item['importance']:.4f}")
        
        # Compile results
        results = {
            'model_type': 'decision_tree',
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'n_features': int(X.shape[1]),
            'n_train_samples': int(X_train.shape[0]),
            'n_test_samples': int(X_test.shape[0]),
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'feature_importances': importance_list,
            'classification_report': classification_report(y_test, y_test_pred, 
                                                          labels=unique_labels,
                                                          target_names=target_names,
                                                          zero_division=0,
                                                          output_dict=True),
            'confusion_matrix': cm.tolist()
        }
        
        return dt_model, results
    
    def train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_mapping: Dict[int, str],
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        Train a random forest classifier and extract feature importances.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0=Normal, 1=Pneumonia)
            feature_mapping: Dictionary mapping feature numbers to names
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            n_estimators: Number of trees in the forest
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in leaf node
            n_jobs: Number of parallel jobs (-1 = use all cores)
            
        Returns:
            - Trained RandomForestClassifier
            - Dictionary containing model metrics and feature importances
        """
        print(f"\n{'='*70}")
        print(f"TRAINING RANDOM FOREST CLASSIFIER")
        print(f"{'='*70}\n")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {X.shape[1]}")
        print(f"Number of trees: {n_estimators}")
        
        # Train random forest
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=n_jobs,
            verbose=1
        )
        
        print("\nTraining model...")
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\n✓ Training Accuracy: {train_accuracy:.4f}")
        print(f"✓ Test Accuracy: {test_accuracy:.4f}")
        
        print("\nClassification Report (Test Set):")
        # Get unique labels present in test set
        unique_labels = sorted(set(y_test) | set(y_test_pred))
        target_names = ['Normal' if l == 0 else 'Pneumonia' for l in unique_labels]
        print(classification_report(y_test, y_test_pred, 
                                   labels=unique_labels,
                                   target_names=target_names,
                                   zero_division=0))
        
        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, y_test_pred, labels=unique_labels)
        print(cm)
        
        # Extract feature importances (averaged across all trees)
        feature_importances = rf_model.feature_importances_
        
        # Create sorted list of (feature_number, feature_name, importance)
        importance_list = []
        for i, importance in enumerate(feature_importances):
            feature_num = i + 1
            feature_name = feature_mapping.get(feature_num, f"feature_{feature_num}")
            importance_list.append({
                'feature_number': feature_num,
                'feature_name': feature_name,
                'importance': float(importance)
            })
        
        # Sort by importance
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        print(f"\nTop 10 Most Important Features (Random Forest):")
        for i, item in enumerate(importance_list[:10], 1):
            print(f"  {i}. Feature {item['feature_number']:3d} "
                  f"({item['feature_name'][:50]}): {item['importance']:.4f}")
        
        # Compile results
        results = {
            'model_type': 'random_forest',
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'n_features': int(X.shape[1]),
            'n_train_samples': int(X_train.shape[0]),
            'n_test_samples': int(X_test.shape[0]),
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'feature_importances': importance_list,
            'classification_report': classification_report(y_test, y_test_pred, 
                                                          labels=unique_labels,
                                                          target_names=target_names,
                                                          zero_division=0,
                                                          output_dict=True),
            'confusion_matrix': cm.tolist()
        }
        
        return rf_model, results
    
    def save_models_and_weights(
        self,
        dt_model: DecisionTreeClassifier,
        dt_results: Dict[str, Any],
        rf_model: RandomForestClassifier,
        rf_results: Dict[str, Any],
        feature_mapping: Dict[int, str],
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Save trained models and their feature importances/weights.
        
        Args:
            dt_model: Trained decision tree model
            dt_results: Decision tree results and metrics
            rf_model: Trained random forest model
            rf_results: Random forest results and metrics
            feature_mapping: Feature number to name mapping
            output_dir: Directory to save outputs (default: self.output_dir)
        """
        save_dir = output_dir or self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"SAVING MODELS AND WEIGHTS")
        print(f"{'='*70}\n")
        
        # Save Decision Tree model
        dt_model_path = os.path.join(save_dir, "decision_tree_model.pkl")
        with open(dt_model_path, 'wb') as f:
            pickle.dump(dt_model, f)
        print(f"✓ Decision Tree model saved: {dt_model_path}")
        
        # Save Random Forest model
        rf_model_path = os.path.join(save_dir, "random_forest_model.pkl")
        with open(rf_model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"✓ Random Forest model saved: {rf_model_path}")
        
        # Save feature mapping
        mapping_path = os.path.join(save_dir, "feature_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(feature_mapping, f, indent=2)
        print(f"✓ Feature mapping saved: {mapping_path}")
        
        # Save Decision Tree results and weights
        dt_results_path = os.path.join(save_dir, "decision_tree_results.json")
        with open(dt_results_path, 'w') as f:
            json.dump(dt_results, f, indent=2)
        print(f"✓ Decision Tree results saved: {dt_results_path}")
        
        # Save Random Forest results and weights
        rf_results_path = os.path.join(save_dir, "random_forest_results.json")
        with open(rf_results_path, 'w') as f:
            json.dump(rf_results, f, indent=2)
        print(f"✓ Random Forest results saved: {rf_results_path}")
        
        # Save feature importances as CSV for easy viewing
        dt_importance_csv = os.path.join(save_dir, "decision_tree_feature_importances.csv")
        with open(dt_importance_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['feature_number', 'feature_name', 'importance'])
            writer.writeheader()
            writer.writerows(dt_results['feature_importances'])
        print(f"✓ Decision Tree importances CSV: {dt_importance_csv}")
        
        rf_importance_csv = os.path.join(save_dir, "random_forest_feature_importances.csv")
        with open(rf_importance_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['feature_number', 'feature_name', 'importance'])
            writer.writeheader()
            writer.writerows(rf_results['feature_importances'])
        print(f"✓ Random Forest importances CSV: {rf_importance_csv}")
        
        print(f"\n✓ All models and weights saved to: {save_dir}")

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
        save_csv: bool = True,
        save_json: bool = True,
        train_models: bool = False,
        labels_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract features from all images in a directory and optionally train models.

        Args:
            directory: Root directory with images
            recursive: If True, traverse subdirectories
            save_csv: If True, save features to CSV
            save_json: If True, save features to JSON
            train_models: If True, train decision tree and random forest models
            labels_file: Path to CSV file with image_name,label columns (required if train_models=True)

        Returns:
            List of feature dictionaries for each image
        """
        print(f"\n{'='*70}")
        print(f"PHASE 3: PNEUMONIA FEATURE EXTRACTION")
        print(f"{'='*70}\n")
        print(f"Processing directory: {directory}")

        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        all_features = []

        # Collect image paths
        if recursive:
            image_paths = [
                os.path.join(root, fname)
                for root, _, files in os.walk(directory)
                for fname in files
                if os.path.splitext(fname)[1].lower() in image_extensions
            ]
        else:
            image_paths = [
                os.path.join(directory, fname)
                for fname in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, fname))
                and os.path.splitext(fname)[1].lower() in image_extensions
            ]

        print(f"Found {len(image_paths)} images\n")

        # Process each image
        for idx, img_path in enumerate(image_paths, 1):
            try:
                print(f"[{idx}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
                features = self.extract_features(img_path)
                all_features.append(features)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        # Save results
        if save_csv:
            csv_path = os.path.join(self.output_dir, "pneumonia_features.csv")
            self._save_features_csv(all_features, csv_path)
            print(f"\n✓ Features saved to: {csv_path}")

        if save_json:
            json_path = os.path.join(self.output_dir, "pneumonia_features.json")
            with open(json_path, "w") as f:
                json.dump(all_features, f, indent=2)
            print(f"✓ Features saved to: {json_path}")

        print(f"\n✓ Processed {len(all_features)}/{len(image_paths)} images successfully")
        
        # Train models if requested
        if train_models:
            if labels_file is None:
                print("\n⚠ Warning: train_models=True but no labels_file provided.")
                print("Skipping model training. Please provide a labels CSV file.")
            elif len(all_features) < 10:
                print(f"\n⚠ Warning: Only {len(all_features)} samples. Need at least 10 for training.")
                print("Skipping model training.")
            else:
                self._train_and_save_models(all_features, labels_file)
        
        return all_features
    
    def _train_and_save_models(
        self,
        features_list: List[Dict[str, Any]],
        labels_file: str,
    ) -> None:
        """
        Train decision tree and random forest models on extracted features.
        
        Args:
            features_list: List of feature dictionaries
            labels_file: Path to CSV with columns: image_name, label (0=Normal, 1=Pneumonia)
        """
        print(f"\n{'='*70}")
        print(f"PREPARING DATA FOR MODEL TRAINING")
        print(f"{'='*70}\n")
        
        # Load labels
        if not os.path.exists(labels_file):
            print(f"✗ Error: Labels file not found: {labels_file}")
            return
        
        labels_dict = {}
        with open(labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row.get('image_name', row.get('filename', ''))
                label = int(row.get('label', row.get('class', -1)))
                if image_name and label in [0, 1]:
                    labels_dict[image_name] = label
        
        print(f"✓ Loaded labels for {len(labels_dict)} images")
        
        # Convert features to numbered format
        X, feature_mapping, image_names = self.convert_features_to_numbered_format(features_list)
        
        # Match labels with features
        y = []
        valid_indices = []
        for i, img_name in enumerate(image_names):
            if img_name in labels_dict:
                y.append(labels_dict[img_name])
                valid_indices.append(i)
            else:
                # Try matching just the filename without path
                base_name = os.path.basename(img_name)
                if base_name in labels_dict:
                    y.append(labels_dict[base_name])
                    valid_indices.append(i)
        
        if len(y) == 0:
            print("✗ Error: No matching labels found for any images!")
            print("Make sure image names in labels file match the image filenames.")
            return
        
        # Filter to valid samples
        X = X[valid_indices]
        y = np.array(y)
        
        print(f"✓ Matched {len(y)} samples with labels")
        print(f"  - Normal (0): {np.sum(y == 0)} samples")
        print(f"  - Pneumonia (1): {np.sum(y == 1)} samples")
        
        if len(y) < 10:
            print(f"\n✗ Error: Only {len(y)} labeled samples. Need at least 10 for training.")
            return
        
        # Train Decision Tree
        dt_model, dt_results = self.train_decision_tree(
            X, y, feature_mapping,
            test_size=0.2,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        # Train Random Forest
        rf_model, rf_results = self.train_random_forest(
            X, y, feature_mapping,
            test_size=0.2,
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        # Save everything
        self.save_models_and_weights(
            dt_model, dt_results,
            rf_model, rf_results,
            feature_mapping
        )

    def _save_features_csv(self, features_list: List[Dict[str, Any]], output_path: str) -> None:
        """Save features to CSV file."""
        if not features_list:
            return

        # Get all unique keys
        all_keys = set()
        for features in features_list:
            all_keys.update(features.keys())

        # Sort keys for consistent column order
        fieldnames = sorted(all_keys)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features_list)


# def main():
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Phase 3: Extract pneumonia features from chest X-rays and train ML models"
#     )
#     parser.add_argument(
#         "path",
#         help="Image file or directory of images",
#     )
#     parser.add_argument(
#         "--output-dir",
#         default="pneumonia_features",
#         help="Output directory for feature files and models (default: pneumonia_features)",
#     )
#     parser.add_argument(
#         "--no-csv",
#         action="store_true",
#         help="Do not save CSV output",
#     )
#     parser.add_argument(
#         "--no-json",
#         action="store_true",
#         help="Do not save JSON output",
#     )
#     parser.add_argument(
#         "--train-models",
#         action="store_true",
#         help="Train decision tree and random forest models",
#     )
#     parser.add_argument(
#         "--labels-file",
#         help="CSV file with image_name,label columns (required for --train-models)",
#     )

#     args = parser.parse_args()

#     extractor = PneumoniaFeatureExtractor(output_dir=args.output_dir)

#     if os.path.isdir(args.path):
#         extractor.process_directory(
#             args.path,
#             recursive=True,
#             save_csv=not args.no_csv,
#             save_json=not args.no_json,
#             train_models=args.train_models,
#             labels_file=args.labels_file,
#         )
#     else:
#         features = extractor.extract_features(args.path)
#         print(f"\nExtracted {len(features)} features from: {os.path.basename(args.path)}")
        
#         # Print summary
#         print("\nFeature Summary:")
#         numeric_count = 0
#         for key, value in sorted(features.items())[:20]:  # Show first 20
#             if isinstance(value, float):
#                 print(f"  {key}: {value:.4f}")
#                 numeric_count += 1
#             else:
#                 print(f"  {key}: {value}")
        
#         total_numeric = sum(1 for v in features.values() if isinstance(v, (int, float, np.integer, np.floating)))
#         print(f"\n... and {total_numeric - 20} more numeric features")
#         print(f"\nTotal numeric features: {total_numeric}")


# if __name__ == "__main__":
#     main()
