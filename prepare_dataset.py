"""
Dataset Preparation Script for Pneumonia Detection

This script:
1. Samples ~200 images from dataset/chest_xray/chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}
2. Maintains class balance (50% normal, 50% pneumonia)
3. Reorganizes images into: dataset/chest_xray/chest_xray/{normal,pneumonia}
4. Generates dataset_info.csv with columns: image_name, original_path, new_path, label, split
5. Creates train/test/val split metadata for reproducibility

Usage:
    python prepare_dataset.py --total-samples 200 --train-ratio 0.7 --test-ratio 0.2 --val-ratio 0.1
    python prepare_dataset.py --dry-run  # Preview without copying files
"""

import os
import csv
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class DatasetPreparator:
    """Prepare and organize chest X-ray dataset for pneumonia detection."""

    def __init__(
        self,
        source_root: str = "dataset/chest_xray/chest_xray",
        target_root: str = "dataset/chest_xray/chest_xray",
        total_samples: int = 200,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize dataset preparator.

        Args:
            source_root: Root directory containing train/test/val folders
            target_root: Root directory for reorganized normal/pneumonia folders
            total_samples: Total number of images to sample
            train_ratio: Proportion of samples from train split
            test_ratio: Proportion of samples from test split
            val_ratio: Proportion of samples from val split
            random_seed: Random seed for reproducibility
        """
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.total_samples = total_samples
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

        # Validate ratios
        ratio_sum = train_ratio + test_ratio + val_ratio
        if abs(ratio_sum - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

        # Set random seed
        random.seed(random_seed)

        # Define source folders
        self.source_folders = {
            "train": {
                "normal": self.source_root / "train" / "NORMAL",
                "pneumonia": self.source_root / "train" / "PNEUMONIA",
            },
            "test": {
                "normal": self.source_root / "test" / "NORMAL",
                "pneumonia": self.source_root / "test" / "PNEUMONIA",
            },
            "val": {
                "normal": self.source_root / "val" / "NORMAL",
                "pneumonia": self.source_root / "val" / "PNEUMONIA",
            },
        }

        # Define target folders
        self.target_folders = {
            "normal": self.target_root / "normal",
            "pneumonia": self.target_root / "pneumonia",
        }

    def scan_source_images(self) -> Dict[str, Dict[str, List[Path]]]:
        """
        Scan source directories and collect all image paths.

        Returns:
            Nested dict: {split: {class: [image_paths]}}
        """
        print("\n" + "=" * 70)
        print("SCANNING SOURCE DIRECTORIES")
        print("=" * 70 + "\n")

        all_images = {"train": {}, "test": {}, "val": {}}
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}

        for split, folders in self.source_folders.items():
            for class_name, folder in folders.items():
                if not folder.exists():
                    print(f"⚠ Warning: Folder not found: {folder}")
                    all_images[split][class_name] = []
                    continue

                images = [
                    f
                    for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in extensions
                ]
                all_images[split][class_name] = images
                print(f"✓ {split}/{class_name}: {len(images)} images")

        # Print summary
        print("\n" + "-" * 70)
        total_normal = sum(len(all_images[s]["normal"]) for s in ["train", "test", "val"])
        total_pneumonia = sum(len(all_images[s]["pneumonia"]) for s in ["train", "test", "val"])
        print(f"Total available: {total_normal + total_pneumonia} images")
        print(f"  Normal: {total_normal}")
        print(f"  Pneumonia: {total_pneumonia}")
        print("-" * 70)

        return all_images

    def sample_images(
        self, all_images: Dict[str, Dict[str, List[Path]]]
    ) -> List[Dict[str, str]]:
        """
        Sample images according to specified ratios and class balance.

        Args:
            all_images: Nested dict from scan_source_images()

        Returns:
            List of dicts with keys: image_path, class_label, split
        """
        print("\n" + "=" * 70)
        print("SAMPLING IMAGES")
        print("=" * 70 + "\n")

        # Calculate samples per split
        train_samples = int(self.total_samples * self.train_ratio)
        test_samples = int(self.total_samples * self.test_ratio)
        val_samples = self.total_samples - train_samples - test_samples

        split_config = {
            "train": train_samples,
            "test": test_samples,
            "val": val_samples,
        }

        print(f"Total samples to select: {self.total_samples}")
        print(f"  Train: {train_samples} ({self.train_ratio * 100:.1f}%)")
        print(f"  Test: {test_samples} ({self.test_ratio * 100:.1f}%)")
        print(f"  Val: {val_samples} ({self.val_ratio * 100:.1f}%)\n")

        sampled = []

        for split, n_samples in split_config.items():
            # Split equally between normal and pneumonia
            n_normal = n_samples // 2
            n_pneumonia = n_samples - n_normal

            print(f"{split.upper()} split:")
            print(f"  Sampling {n_normal} normal + {n_pneumonia} pneumonia = {n_samples} total")

            # Sample normal images
            normal_available = all_images[split].get("normal", [])
            if len(normal_available) < n_normal:
                print(f"  ⚠ Warning: Only {len(normal_available)} normal images available, need {n_normal}")
                n_normal = len(normal_available)

            normal_sample = random.sample(normal_available, n_normal)
            for img_path in normal_sample:
                sampled.append({
                    "image_path": img_path,
                    "class_label": "normal",
                    "split": split,
                })

            # Sample pneumonia images
            pneumonia_available = all_images[split].get("pneumonia", [])
            if len(pneumonia_available) < n_pneumonia:
                print(f"  ⚠ Warning: Only {len(pneumonia_available)} pneumonia images available, need {n_pneumonia}")
                n_pneumonia = len(pneumonia_available)

            pneumonia_sample = random.sample(pneumonia_available, n_pneumonia)
            for img_path in pneumonia_sample:
                sampled.append({
                    "image_path": img_path,
                    "class_label": "pneumonia",
                    "split": split,
                })

            print(f"  ✓ Selected {n_normal + n_pneumonia} images\n")

        print("-" * 70)
        print(f"Total sampled: {len(sampled)} images")
        print("-" * 70)

        return sampled

    def reorganize_dataset(
        self, sampled_images: List[Dict[str, str]], copy_files: bool = True
    ) -> List[Dict[str, str]]:
        """
        Reorganize sampled images into target folder structure.

        Args:
            sampled_images: List from sample_images()
            copy_files: If True, copy files. If False, dry run (no file operations)

        Returns:
            List of dicts with added 'new_path' key
        """
        print("\n" + "=" * 70)
        if copy_files:
            print("REORGANIZING DATASET")
        else:
            print("DRY RUN - PREVIEW REORGANIZATION")
        print("=" * 70 + "\n")

        # Create target directories
        if copy_files:
            for folder in self.target_folders.values():
                folder.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created target directories:")
            print(f"  {self.target_folders['normal']}")
            print(f"  {self.target_folders['pneumonia']}\n")

        # Copy files and track new paths
        dataset_records = []
        for idx, record in enumerate(sampled_images, 1):
            src_path = Path(record["image_path"])
            class_label = record["class_label"]
            split = record["split"]

            # Generate unique filename to avoid collisions
            new_filename = f"{split}_{src_path.name}"
            dest_path = self.target_folders[class_label] / new_filename

            if copy_files:
                shutil.copy2(src_path, dest_path)

            dataset_records.append({
                "image_name": new_filename,
                "original_path": str(src_path),
                "new_path": str(dest_path),
                "label": 0 if class_label == "normal" else 1,
                "class_name": class_label,
                "split": split,
            })

            if idx % 50 == 0 or idx == len(sampled_images):
                print(f"  Processed {idx}/{len(sampled_images)} images...")

        if copy_files:
            print(f"\n✓ Successfully copied {len(dataset_records)} images")
        else:
            print(f"\n✓ Would copy {len(dataset_records)} images (dry run)")

        return dataset_records

    def save_dataset_info(
        self, dataset_records: List[Dict[str, str]], output_path: str = "dataset_info.csv"
    ) -> None:
        """
        Save dataset information to CSV file.

        Args:
            dataset_records: List from reorganize_dataset()
            output_path: Path to output CSV file
        """
        print("\n" + "=" * 70)
        print("SAVING DATASET INFO")
        print("=" * 70 + "\n")

        fieldnames = ["image_name", "original_path", "new_path", "label", "class_name", "split"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset_records)

        print(f"✓ Saved dataset info to: {output_path}")

        # Print summary statistics
        print("\nDataset Summary:")
        print("-" * 70)

        # Group by split and class
        for split in ["train", "test", "val"]:
            split_records = [r for r in dataset_records if r["split"] == split]
            normal_count = sum(1 for r in split_records if r["label"] == 0)
            pneumonia_count = sum(1 for r in split_records if r["label"] == 1)
            print(f"{split.upper():5s}: {len(split_records):3d} total "
                  f"(Normal: {normal_count:3d}, Pneumonia: {pneumonia_count:3d})")

        total = len(dataset_records)
        total_normal = sum(1 for r in dataset_records if r["label"] == 0)
        total_pneumonia = sum(1 for r in dataset_records if r["label"] == 1)
        print("-" * 70)
        print(f"TOTAL: {total:3d} total (Normal: {total_normal:3d}, Pneumonia: {total_pneumonia:3d})")
        print("=" * 70)

    def run(self, dry_run: bool = False) -> List[Dict[str, str]]:
        """
        Execute the complete dataset preparation pipeline.

        Args:
            dry_run: If True, preview without copying files

        Returns:
            List of dataset records
        """
        # Step 1: Scan source directories
        all_images = self.scan_source_images()

        # Step 2: Sample images
        sampled_images = self.sample_images(all_images)

        # Step 3: Reorganize dataset
        dataset_records = self.reorganize_dataset(sampled_images, copy_files=not dry_run)

        # Step 4: Save dataset info
        if not dry_run:
            self.save_dataset_info(dataset_records)
        else:
            print("\n" + "=" * 70)
            print("DRY RUN COMPLETE - No files were copied")
            print("Run without --dry-run to actually copy files")
            print("=" * 70)

        return dataset_records


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Prepare chest X-ray dataset for pneumonia detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source-root",
        type=str,
        default="dataset/chest_xray/chest_xray",
        help="Root directory containing train/test/val folders",
    )

    parser.add_argument(
        "--target-root",
        type=str,
        default="dataset/chest_xray/chest_xray",
        help="Root directory for reorganized normal/pneumonia folders",
    )

    parser.add_argument(
        "--total-samples",
        type=int,
        default=200,
        help="Total number of images to sample",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of samples from train split",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Proportion of samples from test split",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of samples from val split",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="dataset_info.csv",
        help="Output CSV filename for dataset info",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without copying files",
    )

    args = parser.parse_args()

    # Create preparator
    preparator = DatasetPreparator(
        source_root=args.source_root,
        target_root=args.target_root,
        total_samples=args.total_samples,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
    )

    # Run preparation pipeline
    dataset_records = preparator.run(dry_run=args.dry_run)

    print(f"\n✅ Dataset preparation {'preview' if args.dry_run else 'complete'}!")
    print(f"Total images: {len(dataset_records)}")

    if not args.dry_run:
        print(f"\nNext steps:")
        print(f"1. Review dataset_info.csv to verify the dataset")
        print(f"2. Run preprocessing and segmentation: python preprocess_and_segment.py")


if __name__ == "__main__":
    main()
