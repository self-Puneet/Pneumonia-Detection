"""
Model Training Pipeline for Pneumonia Detection

This script trains Decision Tree and Random Forest classifiers on extracted edge-based
features for pneumonia detection.

Features:
1. Loads features.csv with 43 edge-based features
2. Trains Decision Tree classifier
3. Trains Random Forest classifier
4. Extracts and saves feature importances (model weights)
5. Saves trained models as .pkl files
6. Generates comprehensive performance reports

Usage:
    python train_models.py
    python train_models.py --features features.csv --output-dir models_output
    python train_models.py --test-size 0.3 --n-estimators 200
"""

import os
import csv
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


class PneumoniaModelTrainer:
    """Train and evaluate pneumonia detection models."""
    
    def __init__(
        self,
        features_csv: str = "features.csv",
        output_dir: str = "models_output",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize model trainer.
        
        Args:
            features_csv: Path to features CSV file
            output_dir: Directory to save models and reports
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.features_csv = features_csv
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        # Models
        self.dt_model = None
        self.rf_model = None
        
        # Results
        self.dt_results = {}
        self.rf_results = {}
    
    def load_data(self) -> None:
        """Load features from CSV file."""
        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70)
        
        if not os.path.exists(self.features_csv):
            raise FileNotFoundError(f"Features file not found: {self.features_csv}")
        
        self.df = pd.read_csv(self.features_csv)
        
        print(f"✓ Loaded {len(self.df)} samples from {self.features_csv}")
        print(f"  Columns: {len(self.df.columns)}")
        
        # Check class distribution
        label_counts = self.df['label'].value_counts()
        print(f"\nClass distribution:")
        print(f"  Normal (0): {label_counts.get(0, 0)} samples")
        print(f"  Pneumonia (1): {label_counts.get(1, 0)} samples")
        
        # Warn about class imbalance or single class
        if len(label_counts) < 2:
            print(f"\n⚠ WARNING: Dataset contains only one class!")
            print(f"   Binary classification requires samples from both Normal and Pneumonia.")
            print(f"   Model training will continue but results may not be meaningful.")
        
        # Check for missing values
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            print(f"\n⚠ Warning: {missing} missing values detected")
            print("Filling missing values with 0...")
            self.df = self.df.fillna(0)
        
        print("=" * 70 + "\n")
    
    def prepare_data(self) -> None:
        """Split data into train and test sets."""
        print("=" * 70)
        print("PREPARING DATA")
        print("=" * 70)
        
        # Separate features and labels
        metadata_cols = ['image_name', 'label', 'split']
        self.feature_names = [col for col in self.df.columns if col not in metadata_cols]
        
        X = self.df[self.feature_names].values
        y = self.df['label'].values
        
        print(f"Features: {len(self.feature_names)}")
        print(f"Samples: {len(X)}")
        
        # Split data (skip stratification if only one class)
        n_classes = len(np.unique(y))
        if n_classes > 1:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )
        
        print(f"\nTrain/Test Split:")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
        print(f"  Test size: {self.test_size * 100:.0f}%")
        
        # Check class balance in splits
        train_normal = np.sum(self.y_train == 0)
        train_pneumonia = np.sum(self.y_train == 1)
        test_normal = np.sum(self.y_test == 0)
        test_pneumonia = np.sum(self.y_test == 1)
        
        print(f"\nTraining set distribution:")
        print(f"  Normal: {train_normal}, Pneumonia: {train_pneumonia}")
        print(f"Test set distribution:")
        print(f"  Normal: {test_normal}, Pneumonia: {test_pneumonia}")
        
        print("=" * 70 + "\n")
    
    def train_decision_tree(
        self,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5
    ) -> None:
        """
        Train Decision Tree classifier.
        
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in leaf node
        """
        print("=" * 70)
        print("TRAINING DECISION TREE CLASSIFIER")
        print("=" * 70)
        
        print(f"Hyperparameters:")
        print(f"  max_depth: {max_depth}")
        print(f"  min_samples_split: {min_samples_split}")
        print(f"  min_samples_leaf: {min_samples_leaf}")
        print(f"  class_weight: balanced")
        
        # Initialize model
        self.dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        # Train
        print("\nTraining...")
        self.dt_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        self._evaluate_model(self.dt_model, "Decision Tree")
        
        print("=" * 70 + "\n")
    
    def train_random_forest(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        n_jobs: int = -1
    ) -> None:
        """
        Train Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in leaf node
            n_jobs: Number of parallel jobs
        """
        print("=" * 70)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("=" * 70)
        
        print(f"Hyperparameters:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  min_samples_split: {min_samples_split}")
        print(f"  min_samples_leaf: {min_samples_leaf}")
        print(f"  class_weight: balanced")
        print(f"  n_jobs: {n_jobs}")
        
        # Initialize model
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=n_jobs
        )
        
        # Train
        print("\nTraining...")
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        self._evaluate_model(self.rf_model, "Random Forest")
        
        print("=" * 70 + "\n")
    
    def _evaluate_model(self, model, model_name: str) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            model: Trained sklearn model
            model_name: Name for display
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Calculate metrics
        train_acc = accuracy_score(self.y_train, y_train_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(self.y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, zero_division=0)
        
        # Calculate ROC-AUC only if both classes are present
        n_classes = len(np.unique(np.concatenate([self.y_train, self.y_test])))
        if n_classes > 1:
            y_test_proba = model.predict_proba(self.X_test)[:, 1]
            test_auc = roc_auc_score(self.y_test, y_test_proba)
        else:
            test_auc = 0.0
            print(f"\n⚠ Warning: Only one class present in dataset. ROC-AUC cannot be calculated.")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        # Classification report
        class_report = classification_report(
            self.y_test,
            y_test_pred,
            target_names=['Normal', 'Pneumonia'],
            zero_division=0,
            output_dict=True
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train, cv=5, scoring='accuracy'
        )
        
        # Feature importances
        feature_importances = model.feature_importances_
        importance_list = []
        for i, importance in enumerate(feature_importances):
            importance_list.append({
                'feature_number': i + 1,
                'feature_name': self.feature_names[i],
                'importance': float(importance)
            })
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        # Store results
        results = {
            'model_name': model_name,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'test_auc': float(test_auc),
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'feature_importances': importance_list,
            'n_features': len(self.feature_names),
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test)
        }
        
        # Store in class
        if model_name == "Decision Tree":
            self.dt_results = results
        else:
            self.rf_results = results
        
        # Print results
        print(f"\n{model_name} Results:")
        print("-" * 70)
        print(f"Training Accuracy:   {train_acc:.4f}")
        print(f"Test Accuracy:       {test_acc:.4f}")
        print(f"Test Precision:      {test_precision:.4f}")
        print(f"Test Recall:         {test_recall:.4f}")
        print(f"Test F1-Score:       {test_f1:.4f}")
        print(f"Test ROC-AUC:        {test_auc:.4f}")
        print(f"CV Accuracy:         {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Normal  Pneumonia")
        print(f"Actual Normal    {cm[0][0]:6d}  {cm[0][1]:9d}")
        print(f"       Pneumonia {cm[1][0]:6d}  {cm[1][1]:9d}")
        
        print(f"\nClassification Report:")
        print(f"              Precision  Recall  F1-Score  Support")
        for label, metrics in class_report.items():
            if label in ['Normal', 'Pneumonia']:
                print(f"{label:12s}  {metrics['precision']:9.4f}  {metrics['recall']:6.4f}  "
                      f"{metrics['f1-score']:8.4f}  {int(metrics['support']):7d}")
        
        print(f"\nTop 10 Most Important Features:")
        for i, item in enumerate(importance_list[:10], 1):
            print(f"  {i:2d}. {item['feature_name']:50s}  {item['importance']:.4f}")
        
        return results
    
    def save_models(self) -> None:
        """Save trained models to pickle files."""
        print("=" * 70)
        print("SAVING MODELS")
        print("=" * 70)
        
        # Save Decision Tree
        dt_path = self.output_dir / "decision_tree_model.pkl"
        with open(dt_path, 'wb') as f:
            pickle.dump(self.dt_model, f)
        print(f"✓ Decision Tree saved: {dt_path}")
        
        # Save Random Forest
        rf_path = self.output_dir / "random_forest_model.pkl"
        with open(rf_path, 'wb') as f:
            pickle.dump(self.rf_model, f)
        print(f"✓ Random Forest saved: {rf_path}")
        
        # Save feature names
        feature_names_path = self.output_dir / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"✓ Feature names saved: {feature_names_path}")
        
        print("=" * 70 + "\n")
    
    def save_results(self) -> None:
        """Save evaluation results and feature importances."""
        print("=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        # Save Decision Tree results
        dt_results_path = self.output_dir / "decision_tree_results.json"
        with open(dt_results_path, 'w') as f:
            json.dump(self.dt_results, f, indent=2)
        print(f"✓ Decision Tree results: {dt_results_path}")
        
        # Save Random Forest results
        rf_results_path = self.output_dir / "random_forest_results.json"
        with open(rf_results_path, 'w') as f:
            json.dump(self.rf_results, f, indent=2)
        print(f"✓ Random Forest results: {rf_results_path}")
        
        # Save feature importances as CSV
        dt_importance_csv = self.output_dir / "decision_tree_feature_importances.csv"
        with open(dt_importance_csv, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['feature_number', 'feature_name', 'importance']
            )
            writer.writeheader()
            writer.writerows(self.dt_results['feature_importances'])
        print(f"✓ Decision Tree importances CSV: {dt_importance_csv}")
        
        rf_importance_csv = self.output_dir / "random_forest_feature_importances.csv"
        with open(rf_importance_csv, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['feature_number', 'feature_name', 'importance']
            )
            writer.writeheader()
            writer.writerows(self.rf_results['feature_importances'])
        print(f"✓ Random Forest importances CSV: {rf_importance_csv}")
        
        # Save comprehensive training report
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'features_file': self.features_csv,
                'total_samples': len(self.df),
                'n_features': len(self.feature_names),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'test_size': self.test_size,
                'random_state': self.random_state
            },
            'decision_tree': self.dt_results,
            'random_forest': self.rf_results
        }
        
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Comprehensive report: {report_path}")
        
        print("=" * 70 + "\n")
    
    def print_summary(self) -> None:
        """Print final summary comparison."""
        print("=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        
        print("\nModel Comparison:")
        print("-" * 70)
        print(f"{'Metric':<25} {'Decision Tree':>15} {'Random Forest':>15}")
        print("-" * 70)
        
        metrics = [
            ('Test Accuracy', 'test_accuracy'),
            ('Test Precision', 'test_precision'),
            ('Test Recall', 'test_recall'),
            ('Test F1-Score', 'test_f1'),
            ('Test ROC-AUC', 'test_auc'),
            ('CV Mean Accuracy', 'cv_mean_accuracy')
        ]
        
        for metric_name, metric_key in metrics:
            dt_val = self.dt_results[metric_key]
            rf_val = self.rf_results[metric_key]
            print(f"{metric_name:<25} {dt_val:>15.4f} {rf_val:>15.4f}")
        
        print("-" * 70)
        
        # Determine best model
        if self.rf_results['test_accuracy'] > self.dt_results['test_accuracy']:
            best_model = "Random Forest"
            best_acc = self.rf_results['test_accuracy']
        else:
            best_model = "Decision Tree"
            best_acc = self.dt_results['test_accuracy']
        
        print(f"\n🏆 Best Model: {best_model} (Test Accuracy: {best_acc:.4f})")
        
        print("\nOutput Files:")
        print(f"  Models: {self.output_dir}")
        print(f"  - decision_tree_model.pkl")
        print(f"  - random_forest_model.pkl")
        print(f"  - feature_names.json")
        print(f"  - training_report.json")
        print(f"  - *_feature_importances.csv")
        
        print("\n" + "=" * 70)
    
    def run(
        self,
        dt_max_depth: Optional[int] = 10,
        rf_n_estimators: int = 100,
        rf_max_depth: Optional[int] = 15
    ) -> None:
        """
        Run complete training pipeline.
        
        Args:
            dt_max_depth: Decision tree max depth
            rf_n_estimators: Random forest number of trees
            rf_max_depth: Random forest max depth
        """
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Train models
        self.train_decision_tree(max_depth=dt_max_depth)
        self.train_random_forest(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth
        )
        
        # Save everything
        self.save_models()
        self.save_results()
        
        # Print summary
        self.print_summary()


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Train pneumonia detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--features",
        type=str,
        default="features.csv",
        help="Path to features CSV file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models_output",
        help="Directory to save models and results"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--dt-max-depth",
        type=int,
        default=10,
        help="Decision tree maximum depth"
    )
    
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=100,
        help="Random forest number of trees"
    )
    
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=15,
        help="Random forest maximum depth"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PneumoniaModelTrainer(
        features_csv=args.features,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Run training pipeline
    trainer.run(
        dt_max_depth=args.dt_max_depth,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth
    )
    
    print("\n✅ Model training complete!")
    print(f"\nTo use the trained models:")
    print(f"  import pickle")
    print(f"  with open('{args.output_dir}/random_forest_model.pkl', 'rb') as f:")
    print(f"      model = pickle.load(f)")
    print(f"  predictions = model.predict(X_new)")


if __name__ == "__main__":
    main()
