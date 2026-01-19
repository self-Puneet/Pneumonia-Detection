"""
Pneumonia Detection - Prediction Script
========================================
This script takes a new chest X-ray image and predicts whether it shows pneumonia.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --model random_forest
    python predict.py path/to/image.jpg --model decision_tree
"""

import sys
import os
import cv2
import numpy as np
import pickle
import json
from pathlib import Path
import argparse

# Import segmentation
from segmentation import ChestXraySegmenter

# Feature extraction functions (same as in extract_features.py)
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE preprocessing"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return clahe.apply(image)

def get_lung_masks(landmarks, image_shape):
    """Create masks for right and left lungs"""
    right_lung_pts = landmarks[:44]
    left_lung_pts = landmarks[44:94]
    
    right_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    left_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if len(right_lung_pts) > 0:
        cv2.fillPoly(right_mask, [right_lung_pts.astype(np.int32)], 255)
    if len(left_lung_pts) > 0:
        cv2.fillPoly(left_mask, [left_lung_pts.astype(np.int32)], 255)
    
    return right_mask, left_mask

def extract_lung_edge_integrity_features(edges, landmarks, image_shape, prefix):
    """Extract lung edge integrity features"""
    lung_pts = landmarks[:44] if prefix == "RL" else landmarks[44:94]
    
    if len(lung_pts) < 3:
        return {f"{prefix}_edge_completeness": 0, f"{prefix}_edge_smoothness": 0,
                f"{prefix}_edge_avg_magnitude": 0, f"{prefix}_edge_std_magnitude": 0}
    
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [lung_pts.astype(np.int32)], 255)
    boundary = cv2.Canny(mask, 50, 150)
    
    edge_on_boundary = cv2.bitwise_and(edges, boundary)
    completeness = np.sum(edge_on_boundary > 0) / max(np.sum(boundary > 0), 1)
    
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothness = 0
    if contours:
        perimeter = cv2.arcLength(contours[0], True)
        area = cv2.contourArea(contours[0])
        smoothness = (4 * np.pi * area) / max(perimeter**2, 1) if perimeter > 0 else 0
    
    lung_edges = cv2.bitwise_and(edges, mask)
    edge_pixels = lung_edges[lung_edges > 0]
    avg_magnitude = np.mean(edge_pixels) if len(edge_pixels) > 0 else 0
    std_magnitude = np.std(edge_pixels) if len(edge_pixels) > 0 else 0
    
    return {
        f"{prefix}_edge_completeness": completeness,
        f"{prefix}_edge_smoothness": smoothness,
        f"{prefix}_edge_avg_magnitude": avg_magnitude,
        f"{prefix}_edge_std_magnitude": std_magnitude
    }

def extract_edge_loss_features(edges, landmarks, image_shape, prefix):
    """Extract edge loss/discontinuity features"""
    from skimage.morphology import skeletonize
    
    lung_pts = landmarks[:44] if prefix == "RL" else landmarks[44:94]
    
    if len(lung_pts) < 3:
        return {f"{prefix}_num_gaps": 0, f"{prefix}_avg_gap_length": 0,
                f"{prefix}_max_gap_length": 0, f"{prefix}_edge_fragmentation": 0}
    
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [lung_pts.astype(np.int32)], 255)
    boundary = cv2.Canny(mask, 50, 150)
    
    edge_on_boundary = cv2.bitwise_and(edges, boundary)
    boundary_bool = boundary > 0
    edge_bool = edge_on_boundary > 0
    
    gaps = boundary_bool & ~edge_bool
    num_labels, labels = cv2.connectedComponents(gaps.astype(np.uint8))
    
    num_gaps = num_labels - 1
    gap_lengths = []
    for i in range(1, num_labels):
        gap_lengths.append(np.sum(labels == i))
    
    avg_gap_length = np.mean(gap_lengths) if gap_lengths else 0
    max_gap_length = np.max(gap_lengths) if gap_lengths else 0
    
    lung_edges = cv2.bitwise_and(edges, mask)
    skeleton = skeletonize(lung_edges > 0)
    num_edge_labels, _ = cv2.connectedComponents(skeleton.astype(np.uint8))
    fragmentation = num_edge_labels - 1
    
    return {
        f"{prefix}_num_gaps": num_gaps,
        f"{prefix}_avg_gap_length": avg_gap_length,
        f"{prefix}_max_gap_length": max_gap_length,
        f"{prefix}_edge_fragmentation": fragmentation
    }

def extract_opacity_patch_features(image, edges, landmarks, image_shape, prefix):
    """Extract opacity patch features"""
    lung_pts = landmarks[:44] if prefix == "RL" else landmarks[44:94]
    
    if len(lung_pts) < 3:
        return {f"{prefix}_num_opacity_regions": 0, f"{prefix}_opacity_area_ratio": 0,
                f"{prefix}_avg_opacity_intensity": 0, f"{prefix}_patch_irregularity": 0,
                f"{prefix}_avg_patch_size": 0}
    
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [lung_pts.astype(np.int32)], 255)
    
    lung_region = cv2.bitwise_and(image, mask)
    lung_pixels = lung_region[mask > 0]
    
    if len(lung_pixels) == 0:
        return {f"{prefix}_num_opacity_regions": 0, f"{prefix}_opacity_area_ratio": 0,
                f"{prefix}_avg_opacity_intensity": 0, f"{prefix}_patch_irregularity": 0,
                f"{prefix}_avg_patch_size": 0}
    
    threshold = np.percentile(lung_pixels, 75)
    opacity_mask = (lung_region > threshold) & (mask > 0)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opacity_mask.astype(np.uint8))
    num_regions = num_labels - 1
    
    total_lung_area = np.sum(mask > 0)
    opacity_area = np.sum(opacity_mask)
    area_ratio = opacity_area / max(total_lung_area, 1)
    
    avg_intensity = np.mean(lung_region[opacity_mask]) if opacity_area > 0 else 0
    
    irregularities = []
    patch_sizes = []
    for i in range(1, num_labels):
        region_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            area = stats[i, cv2.CC_STAT_AREA]
            patch_sizes.append(area)
            if area > 0 and perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter**2)
                irregularities.append(1 - circularity)
    
    avg_irregularity = np.mean(irregularities) if irregularities else 0
    avg_patch_size = np.mean(patch_sizes) if patch_sizes else 0
    
    return {
        f"{prefix}_num_opacity_regions": num_regions,
        f"{prefix}_opacity_area_ratio": area_ratio,
        f"{prefix}_avg_opacity_intensity": avg_intensity,
        f"{prefix}_patch_irregularity": avg_irregularity,
        f"{prefix}_avg_patch_size": avg_patch_size
    }

def extract_zonal_distribution_features(edges, landmarks, image_shape, prefix):
    """Extract zonal distribution features"""
    lung_pts = landmarks[:44] if prefix == "RL" else landmarks[44:94]
    
    if len(lung_pts) < 3:
        return {f"{prefix}_upper_edge_density": 0, f"{prefix}_middle_edge_density": 0,
                f"{prefix}_lower_edge_density": 0}
    
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [lung_pts.astype(np.int32)], 255)
    
    min_y = int(np.min(lung_pts[:, 1]))
    max_y = int(np.max(lung_pts[:, 1]))
    height = max_y - min_y
    
    third = height // 3
    upper_zone = mask.copy()
    upper_zone[min_y + third:, :] = 0
    
    middle_zone = mask.copy()
    middle_zone[:min_y + third, :] = 0
    middle_zone[min_y + 2*third:, :] = 0
    
    lower_zone = mask.copy()
    lower_zone[:min_y + 2*third, :] = 0
    
    lung_edges = cv2.bitwise_and(edges, mask)
    
    upper_density = np.sum(cv2.bitwise_and(lung_edges, upper_zone) > 0) / max(np.sum(upper_zone > 0), 1)
    middle_density = np.sum(cv2.bitwise_and(lung_edges, middle_zone) > 0) / max(np.sum(middle_zone > 0), 1)
    lower_density = np.sum(cv2.bitwise_and(lung_edges, lower_zone) > 0) / max(np.sum(lower_zone > 0), 1)
    
    return {
        f"{prefix}_upper_edge_density": upper_density,
        f"{prefix}_middle_edge_density": middle_density,
        f"{prefix}_lower_edge_density": lower_density
    }

def extract_silhouette_sign_features(edges, landmarks, image_shape):
    """Extract heart-lung border silhouette features"""
    if len(landmarks) < 94:
        return {"heart_border_sharpness": 0, "heart_edge_continuity": 0, 
                "heart_border_clarity": 0}
    
    left_lung_pts = landmarks[44:94]
    heart_pts = landmarks[94:120]
    
    if len(left_lung_pts) < 3 or len(heart_pts) < 3:
        return {"heart_border_sharpness": 0, "heart_edge_continuity": 0,
                "heart_border_clarity": 0}
    
    left_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    heart_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(left_mask, [left_lung_pts.astype(np.int32)], 255)
    cv2.fillPoly(heart_mask, [heart_pts.astype(np.int32)], 255)
    
    border_region = cv2.bitwise_and(
        cv2.dilate(left_mask, np.ones((5, 5), np.uint8)),
        cv2.dilate(heart_mask, np.ones((5, 5), np.uint8))
    )
    
    border_edges = cv2.bitwise_and(edges, border_region)
    sharpness = np.mean(border_edges[border_region > 0]) if np.sum(border_region > 0) > 0 else 0
    
    heart_boundary = cv2.Canny(heart_mask, 50, 150)
    border_boundary = cv2.bitwise_and(heart_boundary, border_region)
    edge_on_border = cv2.bitwise_and(border_edges, border_boundary)
    continuity = np.sum(edge_on_border > 0) / max(np.sum(border_boundary > 0), 1)
    
    clarity = sharpness * continuity
    
    return {
        "heart_border_sharpness": sharpness,
        "heart_edge_continuity": continuity,
        "heart_border_clarity": clarity
    }

def extract_asymmetry_features(edges, landmarks, image_shape):
    """Extract left-right asymmetry features"""
    if len(landmarks) < 94:
        return {"LR_edge_density_diff": 0, "LR_edge_intensity_diff": 0,
                "LR_opacity_diff": 0, "LR_pattern_correlation": 0}
    
    right_lung_pts = landmarks[:44]
    left_lung_pts = landmarks[44:94]
    
    right_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    left_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if len(right_lung_pts) >= 3:
        cv2.fillPoly(right_mask, [right_lung_pts.astype(np.int32)], 255)
    if len(left_lung_pts) >= 3:
        cv2.fillPoly(left_mask, [left_lung_pts.astype(np.int32)], 255)
    
    right_edges = cv2.bitwise_and(edges, right_mask)
    left_edges = cv2.bitwise_and(edges, left_mask)
    
    right_density = np.sum(right_edges > 0) / max(np.sum(right_mask > 0), 1)
    left_density = np.sum(left_edges > 0) / max(np.sum(left_mask > 0), 1)
    density_diff = abs(right_density - left_density)
    
    right_intensity = np.mean(right_edges[right_edges > 0]) if np.sum(right_edges > 0) > 0 else 0
    left_intensity = np.mean(left_edges[left_edges > 0]) if np.sum(left_edges > 0) > 0 else 0
    intensity_diff = abs(right_intensity - left_intensity)
    
    opacity_diff = abs(np.sum(right_mask > 0) - np.sum(left_mask > 0)) / max(image_shape[0] * image_shape[1], 1)
    
    if np.sum(right_edges > 0) > 0 and np.sum(left_edges > 0) > 0:
        right_flat = right_edges[right_mask > 0].flatten()
        left_flat = left_edges[left_mask > 0].flatten()
        min_len = min(len(right_flat), len(left_flat))
        if min_len > 1:
            correlation = np.corrcoef(right_flat[:min_len], left_flat[:min_len])[0, 1]
            if np.isnan(correlation):
                correlation = 0
        else:
            correlation = 0
    else:
        correlation = 0
    
    return {
        "LR_edge_density_diff": density_diff,
        "LR_edge_intensity_diff": intensity_diff,
        "LR_opacity_diff": opacity_diff,
        "LR_pattern_correlation": correlation
    }

def extract_all_features(image, landmarks):
    """Extract all 43 features from an image"""
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    features = {}
    
    # Right lung features
    rl_edge_features = extract_lung_edge_integrity_features(edges, landmarks, image.shape, "RL")
    features.update(rl_edge_features)
    
    rl_loss_features = extract_edge_loss_features(edges, landmarks, image.shape, "RL")
    features.update(rl_loss_features)
    
    rl_opacity_features = extract_opacity_patch_features(gray, edges, landmarks, image.shape, "RL")
    features.update(rl_opacity_features)
    
    rl_zonal_features = extract_zonal_distribution_features(edges, landmarks, image.shape, "RL")
    features.update(rl_zonal_features)
    
    # Left lung features
    ll_edge_features = extract_lung_edge_integrity_features(edges, landmarks[44:], image.shape, "LL")
    features.update(ll_edge_features)
    
    ll_loss_features = extract_edge_loss_features(edges, landmarks[44:], image.shape, "LL")
    features.update(ll_loss_features)
    
    ll_opacity_features = extract_opacity_patch_features(gray, edges, landmarks[44:], image.shape, "LL")
    features.update(ll_opacity_features)
    
    ll_zonal_features = extract_zonal_distribution_features(edges, landmarks[44:], image.shape, "LL")
    features.update(ll_zonal_features)
    
    # Heart-lung border features
    silhouette_features = extract_silhouette_sign_features(edges, landmarks, image.shape)
    features.update(silhouette_features)
    
    # Asymmetry features
    asymmetry_features = extract_asymmetry_features(edges, landmarks, image.shape)
    features.update(asymmetry_features)
    
    return features

def preprocess_and_segment_image(image_path, segmenter):
    """Preprocess image and extract landmarks"""
    print(f"\n📸 Loading image: {image_path}")
    
    # Run segmentation (it handles loading internally)
    print("   Running segmentation to extract landmarks...")
    result = segmenter.segment(str(image_path))
    landmarks = result['landmarks']
    
    print(f"   Extracted {len(landmarks)} landmarks")
    
    # Load image for feature extraction
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"   Image shape: {image.shape}")
    
    # Apply CLAHE preprocessing
    print("   Applying CLAHE preprocessing...")
    preprocessed = apply_clahe(image)
    
    return preprocessed, landmarks

def predict_pneumonia(image_path, model_path='models_output/random_forest_model.pkl'):
    """
    Main prediction function
    
    Args:
        image_path: Path to chest X-ray image
        model_path: Path to saved model pickle file
        
    Returns:
        dict with prediction results
    """
    print("="*70)
    print("🏥 PNEUMONIA DETECTION SYSTEM")
    print("="*70)
    
    # Initialize segmenter
    print("\n🔧 Initializing segmentation model...")
    segmenter = ChestXraySegmenter(weights_path='weights.pt')
    
    # Preprocess and segment
    preprocessed_img, landmarks = preprocess_and_segment_image(image_path, segmenter)
    
    # Extract features
    print("\n🔬 Extracting 43 edge-based features...")
    features = extract_all_features(preprocessed_img, landmarks)
    
    # Prepare feature vector
    feature_names = [
        'RL_edge_completeness', 'RL_edge_smoothness', 'RL_edge_avg_magnitude', 'RL_edge_std_magnitude',
        'RL_num_gaps', 'RL_avg_gap_length', 'RL_max_gap_length', 'RL_edge_fragmentation',
        'RL_num_opacity_regions', 'RL_opacity_area_ratio', 'RL_avg_opacity_intensity', 
        'RL_patch_irregularity', 'RL_avg_patch_size',
        'RL_upper_edge_density', 'RL_middle_edge_density', 'RL_lower_edge_density',
        'LL_edge_completeness', 'LL_edge_smoothness', 'LL_edge_avg_magnitude', 'LL_edge_std_magnitude',
        'LL_num_gaps', 'LL_avg_gap_length', 'LL_max_gap_length', 'LL_edge_fragmentation',
        'LL_num_opacity_regions', 'LL_opacity_area_ratio', 'LL_avg_opacity_intensity',
        'LL_patch_irregularity', 'LL_avg_patch_size',
        'LL_upper_edge_density', 'LL_middle_edge_density', 'LL_lower_edge_density',
        'heart_border_sharpness', 'heart_edge_continuity', 'heart_border_clarity',
        'LR_edge_density_diff', 'LR_edge_intensity_diff', 'LR_opacity_diff', 'LR_pattern_correlation',
        'RL_sobel_x_mean', 'RL_sobel_y_mean', 'LL_sobel_x_mean', 'LL_sobel_y_mean'
    ]
    
    feature_vector = np.array([[features.get(name, 0) for name in feature_names]])
    
    print(f"   Feature vector shape: {feature_vector.shape}")
    
    # Load model
    print(f"\n🤖 Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"   Model type: {type(model).__name__}")
    
    # Make prediction
    print("\n🔮 Making prediction...")
    prediction = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]
    
    # Get class labels
    class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
    predicted_class = class_names[prediction]
    confidence = probabilities[prediction] * 100
    
    # Prepare results
    results = {
        'image_path': str(image_path),
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': {
            'NORMAL': probabilities[0] * 100,
            'PNEUMONIA': probabilities[1] * 100
        },
        'features': features,
        'model_used': model_path
    }
    
    # Display results
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    print(f"\n🎯 Prediction: {predicted_class}")
    print(f"📈 Confidence: {confidence:.2f}%")
    print(f"\n   Probability of NORMAL:    {probabilities[0]*100:.2f}%")
    print(f"   Probability of PNEUMONIA: {probabilities[1]*100:.2f}%")
    
    # Feature importance (if Random Forest)
    if hasattr(model, 'feature_importances_'):
        print("\n🔍 Top 5 Most Important Features:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        for i, idx in enumerate(indices, 1):
            feature_name = feature_names[idx]
            feature_value = features.get(feature_name, 0)
            importance = importances[idx] * 100
            print(f"   {i}. {feature_name}: {feature_value:.4f} (importance: {importance:.2f}%)")
    
    print("\n" + "="*70)
    
    return results

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Predict pneumonia from chest X-ray image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict.py image.jpg
    python predict.py image.jpg --model models_output/random_forest_model.pkl
    python predict.py image.jpg --model models_output/decision_tree_model.pkl --output result.json
        """
    )
    
    parser.add_argument('image_path', type=str, help='Path to chest X-ray image')
    parser.add_argument('--model', type=str, default='models_output/random_forest_model.pkl',
                       help='Path to saved model file (default: models_output/random_forest_model.pkl)')
    parser.add_argument('--output', type=str, help='Path to save results as JSON (optional)')
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Run prediction
    try:
        results = predict_pneumonia(args.image_path, args.model)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if results['prediction'] == 'NORMAL' else 1)
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == '__main__':
    main()
