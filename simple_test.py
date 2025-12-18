"""
Simple test script for trained pneumonia detection models.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.hybridgnet_landmarks import ChestXrayLandmarkCentralizer
from models.pneumonia_features import PneumoniaFeatureExtractor
from utils.dataset_io import list_chest_xray_images

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def test_models(
    dataset_root="dataset/chest_xray",
    max_samples=200,
    models_dir="pneumonia_features",
    weights_path="weights.pt"
):
    """Test trained models on unseen data."""
    
    print("=" * 70)
    print("TESTING TRAINED PNEUMONIA DETECTION MODELS")
    print("=" * 70 + "\n")
    
    # Load saved models
    print("Loading models...")
    dt_path = os.path.join(models_dir, "decision_tree_model.pkl")
    rf_path = os.path.join(models_dir, "random_forest_model.pkl")
    
    with open(dt_path, 'rb') as f:
        dt_model = pickle.load(f)
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    print("✓ Decision Tree model loaded")
    print("✓ Random Forest model loaded\n")
    
    # Initialize feature extractor
    landmark_model = ChestXrayLandmarkCentralizer(weights_path=weights_path)
    extractor = PneumoniaFeatureExtractor(landmark_centralizer=landmark_model)
    
    # Load test data
    print("Loading test data...")
    image_paths, labels = list_chest_xray_images(dataset_root, shuffle=True)
    
    if max_samples:
        image_paths = image_paths[:max_samples]
        labels = labels[:max_samples]
    
    print(f"Total test images: {len(image_paths)}")
    print(f"  Normal: {labels.count(0)}")
    print(f"  Pneumonia: {labels.count(1)}\n")
    
    # Extract features
    print("Extracting features...")
    features_list = []
    valid_labels = []
    
    for idx, (img_path, label) in enumerate(zip(image_paths, labels), 1):
        try:
            print(f"[{idx}/{len(image_paths)}] {os.path.basename(img_path)}")
            feats = extractor.extract_features(img_path)
            features_list.append(feats)
            valid_labels.append(label)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    X, _, _ = extractor.convert_features_to_numbered_format(features_list)
    y = np.array(valid_labels)
    
    print(f"\n✓ Processed: {len(y)}/{len(image_paths)} images\n")
    
    # Make predictions
    print("Making predictions...")
    dt_pred = dt_model.predict(X)
    rf_pred = rf_model.predict(X)
    
    dt_acc = accuracy_score(y, dt_pred)
    rf_acc = accuracy_score(y, rf_pred)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("DECISION TREE RESULTS:")
    print(f"  Accuracy: {dt_acc:.4f}")
    dt_cm = confusion_matrix(y, dt_pred, labels=[0, 1])
    print(f"  Confusion Matrix:")
    print(f"    [[{dt_cm[0,0]:3d} {dt_cm[0,1]:3d}]")
    print(f"     [{dt_cm[1,0]:3d} {dt_cm[1,1]:3d}]]")
    
    unique_labels = sorted(set(y) | set(dt_pred))
    target_names = ['Normal' if l == 0 else 'Pneumonia' for l in unique_labels]
    print("\n" + classification_report(y, dt_pred, labels=unique_labels, 
                                      target_names=target_names, zero_division=0))
    
    print("\nRANDOM FOREST RESULTS:")
    print(f"  Accuracy: {rf_acc:.4f}")
    rf_cm = confusion_matrix(y, rf_pred, labels=[0, 1])
    print(f"  Confusion Matrix:")
    print(f"    [[{rf_cm[0,0]:3d} {rf_cm[0,1]:3d}]")
    print(f"     [{rf_cm[1,0]:3d} {rf_cm[1,1]:3d}]]")
    
    unique_labels = sorted(set(y) | set(rf_pred))
    target_names = ['Normal' if l == 0 else 'Pneumonia' for l in unique_labels]
    print("\n" + classification_report(y, rf_pred, labels=unique_labels,
                                      target_names=target_names, zero_division=0))
    
    # Model comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"Decision Tree Accuracy: {dt_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    if dt_acc > rf_acc:
        print(f"✓ Decision Tree performs better (+{(dt_acc - rf_acc):.4f})")
    elif rf_acc > dt_acc:
        print(f"✓ Random Forest performs better (+{(rf_acc - dt_acc):.4f})")
    else:
        print("✓ Both models perform equally")
    
    print(f"\n{'=' * 70}")
    print("TESTING COMPLETE!")
    print(f"{'=' * 70}\n")
    
    return {
        'decision_tree_accuracy': float(dt_acc),
        'random_forest_accuracy': float(rf_acc),
        'n_samples': int(len(y)),
        'n_normal': int((y == 0).sum()),
        'n_pneumonia': int((y == 1).sum())
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained models")
    parser.add_argument('--dataset', default='dataset/chest_xray')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--models-dir', default='pneumonia_features')
    parser.add_argument('--weights', default='weights.pt')
    
    args = parser.parse_args()
    
    test_models(
        dataset_root=args.dataset,
        max_samples=args.max_samples,
        models_dir=args.models_dir,
        weights_path=args.weights
    )