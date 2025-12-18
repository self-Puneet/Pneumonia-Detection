from pipeline.run_small_experiment import run_demo

results = run_demo(
    dataset_root="dataset/chest_xray",
    max_samples=100,  # Increased to get both classes
    train_models=True,
    weights_path="weights.pt",
)
