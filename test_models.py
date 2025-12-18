"""
Test trained pneumonia detection models on unseen data.
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.hybridgnet_landmarks import ChestXrayLandmarkCentralizer
from models.pneumonia_features import PneumoniaFeatureExtractor
from utils.dataset_io import list_chest_xray_images

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


"""
Test trained pneumonia detection models on unseen data.
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.hybridgnet_landmarks import ChestXrayLandmarkCentralizer
from models.pneumonia_features import PneumoniaFeatureExtractor
from utils.dataset_io import list_chest_xray_images

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_trained_models(models_dir="pneumonia_features"):
    """Load the saved models."""
    dt_path = os.path.join(models_dir, "decision_tree_model.pkl")
    rf_path = os.path.join(models_dir, "random_forest_model.pkl")
    
    with open(dt_path, 'rb') as f:
        dt_model = pickle.load(f)
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    print("✓ Loaded Decision Tree model")
    print("✓ Loaded Random Forest model\n")
    return dt_model, rf_model


def test_models(
    dataset_root="dataset/chest_xray",
    max_samples=200,
    models_dir="pneumonia_features",
    weights_path="weights.pt",
    output_dir="test_results"
):
    """Test models on unseen data."""
    
    # Load models
    print("=" * 70)
    print("LOADING TRAINED MODELS")
    print("=" * 70 + "\n")
    dt_model, rf_model = load_trained_models(models_dir)
    
    # Initialize feature extractor
    landmark_model = ChestXrayLandmarkCentralizer(weights_path=weights_path)
    extractor = PneumoniaFeatureExtractor(landmark_centralizer=landmark_model)
    
    # Load test data
    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)
    image_paths, labels = list_chest_xray_images(dataset_root, shuffle=True)
    
    if max_samples:
        image_paths = image_paths[:max_samples]
        labels = labels[:max_samples]
    
    print(f"Total test images: {len(image_paths)}")
    print(f"  Normal: {labels.count(0)}")
    print(f"  Pneumonia: {labels.count(1)}\n")
    
    # Extract features
    print("=" * 70)
    print("EXTRACTING FEATURES")
    print("=" * 70)
    
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
    
    X, feature_mapping, _ = extractor.convert_features_to_numbered_format(features_list)
    y = np.array(valid_labels)
    
    print(f"\n✓ Processed: {len(y)}/{len(image_paths)} images\n")
    
    # Make predictions
    print("=" * 70)
    print("TESTING MODELS")
    print("=" * 70 + "\n")
    
    dt_pred = dt_model.predict(X)
    rf_pred = rf_model.predict(X)
    
    dt_acc = accuracy_score(y, dt_pred)
    rf_acc = accuracy_score(y, rf_pred)
    
    # Print results
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
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'n_samples': int(len(y)),
        'n_normal': int((y == 0).sum()),
        'n_pneumonia': int((y == 1).sum()),
        'decision_tree': {
            'accuracy': float(dt_acc),
            'confusion_matrix': dt_cm.tolist()
        },
        'random_forest': {
            'accuracy': float(rf_acc),
            'confusion_matrix': rf_cm.tolist()
        }
    }
    
    with open(os.path.join(output_dir, "test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, cm, title, acc in zip(axes, [dt_cm, rf_cm], 
                                   ['Decision Tree', 'Random Forest'],
                                   [dt_acc, rf_acc]):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f"{title}\nAccuracy: {acc:.2%}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
    plt.close()
    
    print(f"\n{'=' * 70}")
    print(f"✓ Results saved to: {output_dir}/")
    print(f"  - test_results.json")
    print(f"  - confusion_matrices.png")
    print(f"{'=' * 70}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained models")
    parser.add_argument('--dataset', default='dataset/chest_xray')
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--models-dir', default='pneumonia_features')
    parser.add_argument('--weights', default='weights.pt')
    parser.add_argument('--output-dir', default='test_results')
    
    args = parser.parse_args()
    
    test_models(
        dataset_root=args.dataset,
        max_samples=args.max_samples,
        models_dir=args.models_dir,
        weights_path=args.weights,
        output_dir=args.output_dir
    )
    """Test trained models on unseen data and generate analysis."""
    
    def __init__(
        self,
        models_dir: str = "pneumonia_features",
        weights_path: str = "weights.pt"
    ):
        self.models_dir = models_dir
        self.weights_path = weights_path
        
        # Load models and metadata
        self.dt_model = self._load_model("decision_tree_model.pkl")
        self.rf_model = self._load_model("random_forest_model.pkl")
        self.feature_mapping = self._load_feature_mapping()
        
        # Initialize feature extractor
        landmark_model = ChestXrayLandmarkCentralizer(weights_path=weights_path)
        self.extractor = PneumoniaFeatureExtractor(landmark_centralizer=landmark_model)
        
        print(f"✓ Loaded Decision Tree model")
        print(f"✓ Loaded Random Forest model")
        print(f"✓ Feature mapping: {len(self.feature_mapping)} features\n")
    
    def _load_model(self, filename: str):
        """Load a pickled model."""
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _load_feature_mapping(self) -> Dict[int, str]:
        """Load feature mapping from JSON."""
        path = os.path.join(self.models_dir, "feature_mapping.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature mapping not found: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
            # Convert string keys back to integers
            return {int(k): v for k, v in data.items()}
    
    def test_on_dataset(
        self,
        dataset_root: str,
        max_samples: Optional[int] = None,
        save_results: bool = True,
        output_dir: str = "test_results"
    ) -> Dict:
        """
        Test models on a dataset and return comprehensive results.
        
        Args:
            dataset_root: Path to dataset directory with normal/ and pneumonia/
            max_samples: Maximum number of samples to test (None for all)
            save_results: Whether to save results to disk
            output_dir: Directory to save results
            
        Returns:
            Dictionary with test results and predictions
        """
        print("=" * 70)
        print("LOADING TEST DATA")
        print("=" * 70)
        
        # Load test images
        image_paths, labels = list_chest_xray_images(dataset_root, shuffle=True)
        
        if max_samples:
            image_paths = image_paths[:max_samples]
            labels = labels[:max_samples]
        
        print(f"Total test images: {len(image_paths)}")
        print(f"  Normal: {labels.count(0)}")
        print(f"  Pneumonia: {labels.count(1)}\n")
        
        # Extract features
        print("=" * 70)
        print("EXTRACTING FEATURES")
        print("=" * 70)
        
        features_list = []
        valid_labels = []
        valid_paths = []
        
        for idx, (img_path, label) in enumerate(zip(image_paths, labels), 1):
            try:
                print(f"[{idx}/{len(image_paths)}] {os.path.basename(img_path)}")
                feats = self.extractor.extract_features(img_path)
                features_list.append(feats)
                valid_labels.append(label)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        
        # Convert to feature matrix
        X, _, image_names = self.extractor.convert_features_to_numbered_format(features_list)
        y = np.array(valid_labels)
        
        print(f"\nSuccessfully processed: {len(y)}/{len(image_paths)} images\n")
        
        # Make predictions
        print("=" * 70)
        print("MAKING PREDICTIONS")
        print("=" * 70)
        
        dt_pred = self.dt_model.predict(X)
        rf_pred = self.rf_model.predict(X)
        
        # Get probability predictions for ROC curve
        dt_proba = self.dt_model.predict_proba(X)[:, 1] if hasattr(self.dt_model, 'predict_proba') else None
        rf_proba = self.rf_model.predict_proba(X)[:, 1] if hasattr(self.rf_model, 'predict_proba') else None
        
        # Compute metrics
        results = {
            'n_samples': len(y),
            'n_normal': int((y == 0).sum()),
            'n_pneumonia': int((y == 1).sum()),
            'decision_tree': self._compute_metrics(y, dt_pred, dt_proba, "Decision Tree"),
            'random_forest': self._compute_metrics(y, rf_pred, rf_proba, "Random Forest"),
            'per_image_predictions': []
        }
        
        # Store per-image predictions
        for i, (path, true_label, dt_p, rf_p) in enumerate(zip(valid_paths, y, dt_pred, rf_pred)):
            results['per_image_predictions'].append({
                'image': os.path.basename(path),
                'true_label': 'Normal' if true_label == 0 else 'Pneumonia',
                'dt_prediction': 'Normal' if dt_p == 0 else 'Pneumonia',
                'rf_prediction': 'Normal' if rf_p == 0 else 'Pneumonia',
                'dt_correct': bool(dt_p == true_label),
                'rf_correct': bool(rf_p == true_label),
            })
        
        # Print detailed analysis
        self._print_analysis(results)
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            self._save_results(results, output_dir)
            self._plot_confusion_matrices(results, output_dir)
            self._plot_feature_importance_comparison(output_dir)
            print(f"\n✓ Results saved to: {output_dir}/")
        
        return results
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        model_name: str
    ) -> Dict:
        """Compute comprehensive metrics for a model."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Get unique labels for classification report
        unique_labels = sorted(set(y_true) | set(y_pred))
        target_names = ['Normal' if l == 0 else 'Pneumonia' for l in unique_labels]
        
        report = classification_report(
            y_true, y_pred,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )
        
        metrics = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'per_class': {
                'normal': {
                    'precision': float(precision[0]) if len(precision) > 0 else 0.0,
                    'recall': float(recall[0]) if len(recall) > 0 else 0.0,
                    'f1': float(f1[0]) if len(f1) > 0 else 0.0,
                    'support': int(support[0]) if len(support) > 0 else 0
                },
                'pneumonia': {
                    'precision': float(precision[1]) if len(precision) > 1 else 0.0,
                    'recall': float(recall[1]) if len(recall) > 1 else 0.0,
                    'f1': float(f1[1]) if len(f1) > 1 else 0.0,
                    'support': int(support[1]) if len(support) > 1 else 0
                }
            }
        }
        
        # Calculate specificity and sensitivity
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        # ROC AUC if probabilities available
        if y_proba is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                metrics['roc_auc'] = float(roc_auc)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def _print_analysis(self, results: Dict):
        """Print detailed analysis of test results."""
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"\nDataset:")
        print(f"  Total samples: {results['n_samples']}")
        print(f"  Normal: {results['n_normal']}")
        print(f"  Pneumonia: {results['n_pneumonia']}")
        
        for model_key in ['decision_tree', 'random_forest']:
            metrics = results[model_key]
            print(f"\n{'-' * 70}")
            print(f"{metrics['model_name'].upper()}")
            print(f"{'-' * 70}")
            print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
            print(f"Sensitivity (Recall for Pneumonia): {metrics['sensitivity']:.4f}")
            print(f"Specificity (Recall for Normal): {metrics['specificity']:.4f}")
            
            if metrics.get('roc_auc'):
                print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            
            print(f"\nPer-Class Metrics:")
            for class_name in ['normal', 'pneumonia']:
                cls_metrics = metrics['per_class'][class_name]
                print(f"  {class_name.capitalize()}:")
                print(f"    Precision: {cls_metrics['precision']:.4f}")
                print(f"    Recall:    {cls_metrics['recall']:.4f}")
                print(f"    F1-Score:  {cls_metrics['f1']:.4f}")
                print(f"    Support:   {cls_metrics['support']}")
            
            print(f"\nConfusion Matrix:")
            cm = np.array(metrics['confusion_matrix'])
            print(f"                Predicted")
            print(f"              Normal  Pneumonia")
            print(f"Actual Normal    {cm[0,0]:3d}      {cm[0,1]:3d}")
            print(f"    Pneumonia    {cm[1,0]:3d}      {cm[1,1]:3d}")
        
        # Compare models
        print(f"\n{'-' * 70}")
        print("MODEL COMPARISON")
        print(f"{'-' * 70}")
        dt_acc = results['decision_tree']['accuracy']
        rf_acc = results['random_forest']['accuracy']
        print(f"Decision Tree Accuracy: {dt_acc:.4f}")
        print(f"Random Forest Accuracy: {rf_acc:.4f}")
        
        if dt_acc > rf_acc:
            print(f"✓ Decision Tree performs better (+{(dt_acc - rf_acc):.4f})")
        elif rf_acc > dt_acc:
            print(f"✓ Random Forest performs better (+{(rf_acc - dt_acc):.4f})")
        else:
            print("✓ Both models perform equally")
        
        # Misclassified images
        print(f"\n{'-' * 70}")
        print("MISCLASSIFIED IMAGES")
        print(f"{'-' * 70}")
        
        dt_errors = [p for p in results['per_image_predictions'] if not p['dt_correct']]
        rf_errors = [p for p in results['per_image_predictions'] if not p['rf_correct']]
        
        print(f"\nDecision Tree Errors: {len(dt_errors)}")
        for err in dt_errors[:10]:  # Show first 10
            print(f"  {err['image']}: True={err['true_label']}, Predicted={err['dt_prediction']}")
        if len(dt_errors) > 10:
            print(f"  ... and {len(dt_errors) - 10} more")
        
        print(f"\nRandom Forest Errors: {len(rf_errors)}")
        for err in rf_errors[:10]:  # Show first 10
            print(f"  {err['image']}: True={err['true_label']}, Predicted={err['rf_prediction']}")
        if len(rf_errors) > 10:
            print(f"  ... and {len(rf_errors) - 10} more")
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save results to JSON file."""
        output_path = os.path.join(output_dir, "test_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Test results saved: {output_path}")
    
    def _plot_confusion_matrices(self, results: Dict, output_dir: str):
        """Plot confusion matrices for both models."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, (model_key, ax) in enumerate(zip(['decision_tree', 'random_forest'], axes)):
            metrics = results[model_key]
            cm = np.array(metrics['confusion_matrix'])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'],
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f"{metrics['model_name']}\nAccuracy: {metrics['accuracy']:.2%}")
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "confusion_matrices.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrices saved: {output_path}")
    
    def _plot_feature_importance_comparison(self, output_dir: str):
        """Plot top feature importances from saved models."""
        try:
            # Load saved results
            dt_results_path = os.path.join(self.models_dir, "decision_tree_results.json")
            rf_results_path = os.path.join(self.models_dir, "random_forest_results.json")
            
            with open(dt_results_path, 'r') as f:
                dt_results = json.load(f)
            with open(rf_results_path, 'r') as f:
                rf_results = json.load(f)
            
            # Get top 10 features
            dt_importance = sorted(
                dt_results['feature_importances'],
                key=lambda x: x['importance'],
                reverse=True
            )[:10]
            
            rf_importance = sorted(
                rf_results['feature_importances'],
                key=lambda x: x['importance'],
                reverse=True
            )[:10]
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Decision Tree
            features = [f"{f['feature_number']}: {f['feature_name'][:30]}" for f in dt_importance]
            importances = [f['importance'] for f in dt_importance]
            axes[0].barh(features, importances, color='skyblue')
            axes[0].set_xlabel('Importance')
            axes[0].set_title('Decision Tree - Top 10 Features')
            axes[0].invert_yaxis()
            
            # Random Forest
            features = [f"{f['feature_number']}: {f['feature_name'][:30]}" for f in rf_importance]
            importances = [f['importance'] for f in rf_importance]
            axes[1].barh(features, importances, color='lightcoral')
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Random Forest - Top 10 Features')
            axes[1].invert_yaxis()
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, "feature_importance_comparison.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Feature importance plot saved: {output_path}")
        except Exception as e:
            print(f"Warning: Could not plot feature importance: {e}")


def main():
    """Run model testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test trained pneumonia detection models on unseen data"
    )
    parser.add_argument(
        '--dataset',
        default='dataset/chest_xray',
        help='Path to test dataset directory (default: dataset/chest_xray)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=200,
        help='Maximum number of test samples (default: 200, use 0 for all)'
    )
    parser.add_argument(
        '--models-dir',
        default='pneumonia_features',
        help='Directory containing trained models (default: pneumonia_features)'
    )
    parser.add_argument(
        '--weights',
        default='weights.pt',
        help='Path to HybridGNet weights (default: weights.pt)'
    )
    parser.add_argument(
        '--output-dir',
        default='test_results',
        help='Output directory for results (default: test_results)'
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = PneumoniaModelTester(
        models_dir=args.models_dir,
        weights_path=args.weights
    )
    
    # Run testing
    max_samples = None if args.max_samples == 0 else args.max_samples
    results = tester.test_on_dataset(
        dataset_root=args.dataset,
        max_samples=max_samples,
        save_results=True,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
