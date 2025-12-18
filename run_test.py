"""Test trained models on unseen data."""

from test_models import test_models

results = test_models(
    dataset_root="dataset/chest_xray",
    max_samples=10,
    models_dir="pneumonia_features",
    weights_path="weights.pt",
    output_dir="test_results"
)
