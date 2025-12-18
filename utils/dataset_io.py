# utils/dataset_io.py

import os
import random
from typing import List, Tuple


def list_chest_xray_images(root: str, shuffle: bool = True, seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Scan dataset/chest_xray/{normal,pneumonia} and return image paths + labels.
    label: 0 = normal, 1 = pneumonia
    
    Args:
        root: Root directory containing normal/ and pneumonia/ folders
        shuffle: Whether to shuffle the images (default: True)
        seed: Random seed for reproducibility (default: 42)
    """
    normal_dir = os.path.join(root, "normal")
    pneu_dir = os.path.join(root, "pneumonia")

    image_paths: List[str] = []
    labels: List[int] = []

    for fname in os.listdir(normal_dir):
        fpath = os.path.join(normal_dir, fname)
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in [".png", ".jpg", ".jpeg"]:
            image_paths.append(fpath)
            labels.append(0)

    for fname in os.listdir(pneu_dir):
        fpath = os.path.join(pneu_dir, fname)
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in [".png", ".jpg", ".jpeg"]:
            image_paths.append(fpath)
            labels.append(1)

    # Shuffle to mix normal and pneumonia cases
    if shuffle:
        combined = list(zip(image_paths, labels))
        random.seed(seed)
        random.shuffle(combined)
        image_paths, labels = zip(*combined) if combined else ([], [])
        image_paths = list(image_paths)
        labels = list(labels)

    return image_paths, labels
