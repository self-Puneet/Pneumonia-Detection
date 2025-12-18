# pipeline/run_small_experiment.py

import os
from typing import List, Dict, Any

import numpy as np

from models.hybridgnet_landmarks import ChestXrayLandmarkCentralizer
from models.pneumonia_features import PneumoniaFeatureExtractor
from utils.dataset_io import list_chest_xray_images


def run_demo(
    dataset_root: str = "dataset/chest_xray",
    max_samples: int = 20,
    train_models: bool = True,
    weights_path: str = None,
) -> Dict[str, Any]:
    """
    Run end-to-end pipeline on a small subset of the dataset.

    Returns dictionary with:
        - 'features_list'
        - 'X'
        - 'y'
        - 'feature_mapping'
        - 'dt_model' (optional)
        - 'rf_model' (optional)
    """
    image_paths, labels = list_chest_xray_images(dataset_root)

    if not image_paths:
        raise RuntimeError(f"No images found in {dataset_root}")

    if max_samples is not None and max_samples > 0:
        image_paths = image_paths[:max_samples]
        labels = labels[:max_samples]

    landmark_model = ChestXrayLandmarkCentralizer(weights_path=weights_path)
    extractor = PneumoniaFeatureExtractor(landmark_centralizer=landmark_model)

    features_list: List[Dict[str, Any]] = []
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Extracting features from {os.path.basename(img_path)}")
        feats = extractor.extract_features(img_path)
        features_list.append(feats)

    X, feature_mapping, image_names = extractor.convert_features_to_numbered_format(features_list)
    y = np.array(labels[: X.shape[0]])

    results: Dict[str, Any] = {
        "features_list": features_list,
        "X": X,
        "y": y,
        "feature_mapping": feature_mapping,
        "image_names": image_names,
    }

    if train_models and len(y) >= 10:
        dt_model, dt_results = extractor.train_decision_tree(
            X, y, feature_mapping,
            test_size=0.2,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
        )

        rf_model, rf_results = extractor.train_random_forest(
            X, y, feature_mapping,
            test_size=0.2,
            n_estimators=50,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
        )

        extractor.save_models_and_weights(
            dt_model, dt_results,
            rf_model, rf_results,
            feature_mapping,
        )

        results["dt_model"] = dt_model
        results["rf_model"] = rf_model
        results["dt_results"] = dt_results
        results["rf_results"] = rf_results

    return results
