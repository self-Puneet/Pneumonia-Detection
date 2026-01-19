"""
Preprocessing and Segmentation Pipeline for Pneumonia Detection

This script:
1. Reads images from dataset/chest_xray/chest_xray/{normal,pneumonia}/
2. Applies preprocessing: CLAHE, resizing, normalization
3. Runs HybridGNet segmentation to extract 120 anatomical landmarks
4. Saves preprocessed images and landmark CSVs to preprocessed_data/
5. Generates a preprocessing report with success/failure statistics

Usage:
    python preprocess_and_segment.py
    python preprocess_and_segment.py --dataset-csv dataset_info.csv --output-dir preprocessed_data
    python preprocess_and_segment.py --skip-preprocessing  # Only run segmentation
    python preprocess_and_segment.py --max-images 50  # Process only first 50 images
"""

import os
import csv
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import cv2
import numpy as np

from segmentation import ChestXraySegmenter


class PreprocessingPipeline:
    """Preprocess chest X-rays and extract anatomical landmarks."""

    def __init__(
        self,
        dataset_csv: str = "dataset_info.csv",
        output_dir: str = "preprocessed_data",
        weights_path: str = "weights.pt",
        apply_clahe: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            dataset_csv: Path to dataset_info.csv from prepare_dataset.py
            output_dir: Directory to save preprocessed images and landmarks
            weights_path: Path to HybridGNet model weights
            apply_clahe: Whether to apply CLAHE contrast enhancement
            target_size: Target image size (width, height). None = keep original
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.dataset_csv = dataset_csv
        self.output_dir = Path(output_dir)
        self.weights_path = weights_path
        self.apply_clahe = apply_clahe
        self.target_size = target_size
        self.normalize = normalize

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.landmarks_dir = self.output_dir / "landmarks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.landmarks_dir.mkdir(parents=True, exist_ok=True)

        # Initialize segmenter
        print(f"Loading segmentation model from: {weights_path}")
        self.segmenter = ChestXraySegmenter(weights_path=weights_path)
        print("✓ Model loaded successfully\n")

        # Statistics
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "failed_images": [],
        }

    def load_dataset_info(self) -> List[Dict[str, str]]:
        """
        Load dataset information from CSV.

        Returns:
            List of dataset records
        """
        if not os.path.exists(self.dataset_csv):
            raise FileNotFoundError(f"Dataset CSV not found: {self.dataset_csv}")

        records = []
        with open(self.dataset_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        print(f"Loaded {len(records)} images from {self.dataset_csv}\n")
        return records

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to an image.

        Args:
            image: Input grayscale image (numpy array)

        Returns:
            Preprocessed image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

        # Resize if target size specified
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] range
        if self.normalize:
            image = image.astype(np.float32) / 255.0
            # Convert back to uint8 for saving
            image = (image * 255).astype(np.uint8)

        return image

    def process_single_image(
        self, image_path: str, image_name: str, label: int, split: str
    ) -> Dict[str, any]:
        """
        Process a single image: preprocess and segment.

        Args:
            image_path: Path to input image
            image_name: Name of the image
            label: Class label (0=normal, 1=pneumonia)
            split: Data split (train/test/val)

        Returns:
            Dictionary with processing results
        """
        result = {
            "image_name": image_name,
            "label": label,
            "split": split,
            "success": False,
            "error": None,
            "preprocessed_path": None,
            "landmarks_path": None,
        }

        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Preprocess
            preprocessed = self.preprocess_image(image)

            # Save preprocessed image
            preprocessed_filename = f"{Path(image_name).stem}_preprocessed.png"
            preprocessed_path = self.images_dir / preprocessed_filename
            cv2.imwrite(str(preprocessed_path), preprocessed)
            result["preprocessed_path"] = str(preprocessed_path)

            # Run segmentation (on original image for best results)
            # The segmenter handles its own preprocessing internally
            segmentation_result = self.segmenter.segment(image_path)
            landmarks = segmentation_result["landmarks"]

            # Save landmarks as CSV
            landmarks_filename = f"{Path(image_name).stem}_landmarks.csv"
            landmarks_path = self.landmarks_dir / landmarks_filename
            self.save_landmarks_csv(landmarks, landmarks_path)
            result["landmarks_path"] = str(landmarks_path)

            result["success"] = True
            self.stats["success"] += 1

        except Exception as e:
            result["error"] = str(e)
            self.stats["failed"] += 1
            self.stats["failed_images"].append({
                "image_name": image_name,
                "error": str(e)
            })

        return result

    def save_landmarks_csv(self, landmarks: np.ndarray, output_path: Path) -> None:
        """
        Save landmarks to CSV file in the format expected by feature extraction.

        Args:
            landmarks: Array of shape (120, 2) with x, y coordinates
            output_path: Path to save CSV
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "x", "y", "landmark_index"])

            # Right lung: landmarks 0-43
            for i in range(44):
                x, y = landmarks[i]
                writer.writerow(["RL", int(x), int(y), i])

            # Left lung: landmarks 44-93
            for i in range(44, 94):
                x, y = landmarks[i]
                writer.writerow(["LL", int(x), int(y), i - 44])

            # Heart: landmarks 94-119
            for i in range(94, 120):
                x, y = landmarks[i]
                writer.writerow(["H", int(x), int(y), i - 94])

    def run(self, max_images: Optional[int] = None) -> List[Dict[str, any]]:
        """
        Run the complete preprocessing and segmentation pipeline.

        Args:
            max_images: Maximum number of images to process (None = all)

        Returns:
            List of processing results
        """
        print("=" * 70)
        print("PREPROCESSING AND SEGMENTATION PIPELINE")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print(f"CLAHE enabled: {self.apply_clahe}")
        print(f"Normalization: {self.normalize}")
        if self.target_size:
            print(f"Target size: {self.target_size}")
        print("=" * 70 + "\n")

        # Load dataset info
        dataset_records = self.load_dataset_info()

        # Limit number of images if specified
        if max_images is not None:
            dataset_records = dataset_records[:max_images]
            print(f"Processing first {max_images} images\n")

        self.stats["total"] = len(dataset_records)

        # Process each image
        results = []
        for idx, record in enumerate(dataset_records, 1):
            image_path = record["new_path"]
            image_name = record["image_name"]
            label = int(record["label"])
            split = record["split"]

            print(f"[{idx}/{len(dataset_records)}] Processing: {image_name}")

            result = self.process_single_image(image_path, image_name, label, split)
            results.append(result)

            if result["success"]:
                print(f"  ✓ Success")
            else:
                print(f"  ✗ Failed: {result['error']}")

        # Save processing report
        self.save_report(results)

        # Print summary
        self.print_summary()

        return results

    def save_report(self, results: List[Dict[str, any]]) -> None:
        """
        Save preprocessing report to JSON and CSV files.

        Args:
            results: List of processing results
        """
        # Save detailed JSON report
        report_json = self.output_dir / "preprocessing_report.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "dataset_csv": self.dataset_csv,
                "weights_path": self.weights_path,
                "apply_clahe": self.apply_clahe,
                "target_size": self.target_size,
                "normalize": self.normalize,
            },
            "statistics": self.stats,
            "results": results,
        }

        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n✓ Detailed report saved to: {report_json}")

        # Save summary CSV
        report_csv = self.output_dir / "preprocessing_summary.csv"
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["image_name", "label", "split", "success", "error", 
                         "preprocessed_path", "landmarks_path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"✓ Summary CSV saved to: {report_csv}")

    def print_summary(self) -> None:
        """Print processing summary statistics."""
        print("\n" + "=" * 70)
        print("PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Total images: {self.stats['total']}")
        print(f"Successfully processed: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")

        if self.stats['failed'] > 0:
            print(f"\nFailed images:")
            for failed in self.stats['failed_images']:
                print(f"  - {failed['image_name']}: {failed['error']}")

        success_rate = (self.stats['success'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")
        print("=" * 70)


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Preprocess chest X-rays and extract anatomical landmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-csv",
        type=str,
        default="dataset_info.csv",
        help="Path to dataset_info.csv from prepare_dataset.py",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="preprocessed_data",
        help="Directory to save preprocessed images and landmarks",
    )

    parser.add_argument(
        "--weights-path",
        type=str,
        default="weights.pt",
        help="Path to HybridGNet model weights",
    )

    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE contrast enhancement",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable pixel normalization",
    )

    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize images to specified size (width height)",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to process (for testing)",
    )

    args = parser.parse_args()

    # Convert target size to tuple if specified
    target_size = tuple(args.target_size) if args.target_size else None

    # Create pipeline
    pipeline = PreprocessingPipeline(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        weights_path=args.weights_path,
        apply_clahe=not args.no_clahe,
        target_size=target_size,
        normalize=not args.no_normalize,
    )

    # Run pipeline
    results = pipeline.run(max_images=args.max_images)

    print(f"\n✅ Preprocessing and segmentation complete!")
    print(f"Preprocessed images: {pipeline.images_dir}")
    print(f"Landmark CSVs: {pipeline.landmarks_dir}")
    print(f"\nNext steps:")
    print(f"1. Review preprocessing_report.json for detailed results")
    print(f"2. Run feature extraction: python extract_features.py")


if __name__ == "__main__":
    main()
