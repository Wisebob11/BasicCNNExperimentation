"""
Enhanced Training Script with Model Comparison

This script allows you to train and compare the original 2-layer CNN
with the enhanced 9-layer CNN using the exact same CIFAR-10 data.
This ensures a fair comparison of model performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

from cnn_model import SimpleCNN
from enhanced_cnn import EnhancedCNN
from data_utils import load_cifar10_data, get_class_distribution
from visualization import plot_training_history, plot_confusion_matrix, plot_sample_predictions


def train_model_comparison(epochs: int = 10, 
                          batch_size: int = 32,
                          validation_split: float = 0.1,
                          save_models: bool = True) -> dict:
    """
    Train both models with identical data for fair comparison.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data for validation
        save_models: Whether to save the trained models
        
    Returns:
        Dictionary containing results for both models
    """
    print("🔥 Enhanced CNN vs Original CNN Comparison")
    print("=" * 60)
    
    # 1. Load and preprocess data (SAME DATA FOR BOTH MODELS)
    print("📁 Loading CIFAR-10 dataset...")
    np.random.seed(42)  # Set seed for reproducible train/val split
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(validation_split)
    
    print(f"\nDataset loaded successfully!")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    
    results = {}
    
    # 2. Train Original 2-Layer CNN
    print(f"\n🚀 Training Original 2-Layer CNN...")
    print("-" * 40)
    
    start_time = time.time()
    
    original_cnn = SimpleCNN()
    original_cnn.build_model()
    print("Original Model Architecture:")
    original_cnn.get_model_summary()
    
    original_history = original_cnn.train(
        x_train, y_train,
        x_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    original_train_time = time.time() - start_time
    
    # Evaluate original model
    original_loss, original_accuracy = original_cnn.evaluate(x_test, y_test)
    original_predictions = original_cnn.predict(x_test)
    
    results['original'] = {
        'model': original_cnn,
        'history': original_history,
        'test_loss': original_loss,
        'test_accuracy': original_accuracy,
        'predictions': original_predictions,
        'train_time': original_train_time,
        'parameters': original_cnn.model.count_params()
    }
    
    print(f"\n✅ Original CNN Results:")
    print(f"  Test Accuracy: {original_accuracy:.4f} ({original_accuracy*100:.2f}%)")
    print(f"  Test Loss: {original_loss:.4f}")
    print(f"  Training Time: {original_train_time:.2f} seconds")
    print(f"  Parameters: {results['original']['parameters']:,}")
    
    # 3. Train Enhanced 9-Layer CNN
    print(f"\n🚀 Training Enhanced 9-Layer CNN...")
    print("-" * 40)
    
    start_time = time.time()
    
    enhanced_cnn = EnhancedCNN()
    enhanced_cnn.build_model()
    print("Enhanced Model Architecture:")
    enhanced_cnn.get_model_summary()
    
    enhanced_history = enhanced_cnn.train(
        x_train, y_train,
        x_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    enhanced_train_time = time.time() - start_time
    
    # Evaluate enhanced model
    enhanced_loss, enhanced_accuracy = enhanced_cnn.evaluate(x_test, y_test)
    enhanced_predictions = enhanced_cnn.predict(x_test)
    
    results['enhanced'] = {
        'model': enhanced_cnn,
        'history': enhanced_history,
        'test_loss': enhanced_loss,
        'test_accuracy': enhanced_accuracy,
        'predictions': enhanced_predictions,
        'train_time': enhanced_train_time,
        'parameters': enhanced_cnn.model.count_params()
    }
    
    print(f"\n✅ Enhanced CNN Results:")
    print(f"  Test Accuracy: {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)")
    print(f"  Test Loss: {enhanced_loss:.4f}")
    print(f"  Training Time: {enhanced_train_time:.2f} seconds")
    print(f"  Parameters: {results['enhanced']['parameters']:,}")
    
    # 4. Comparison Summary
    print(f"\n📊 Model Comparison Summary:")
    print("=" * 60)
    accuracy_improvement = enhanced_accuracy - original_accuracy
    speed_ratio = enhanced_train_time / original_train_time
    param_ratio = results['enhanced']['parameters'] / results['original']['parameters']
    
    print(f"🎯 Accuracy Improvement: +{accuracy_improvement:.4f} ({accuracy_improvement*100:+.2f}%)")
    print(f"⚡ Speed Comparison: {speed_ratio:.1f}x slower (expected due to more layers)")
    print(f"📈 Parameter Increase: {param_ratio:.1f}x more parameters")
    
    if accuracy_improvement > 0:
        print(f"✅ Enhanced model performs BETTER by {accuracy_improvement*100:.2f} percentage points!")
    else:
        print(f"⚠️  Enhanced model performs worse by {abs(accuracy_improvement)*100:.2f} percentage points")
    
    # 5. Create comparison visualizations
    print(f"\n📈 Creating comparison visualizations...")
    create_comparison_plots(results, x_test, y_test)
    
    # 6. Save models if requested
    if save_models:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("models", exist_ok=True)
        
        original_path = f"models/original_2layer_cnn_{timestamp}"
        enhanced_path = f"models/enhanced_9layer_cnn_{timestamp}"
        
        original_cnn.save_model(original_path)
        enhanced_cnn.save_model(enhanced_path)
        
        print(f"\n💾 Models saved:")
        print(f"  Original: {original_path}")
        print(f"  Enhanced: {enhanced_path}")
    
    # 7. Save detailed comparison report
    save_comparison_report(results, epochs, batch_size)
    
    print(f"\n🎉 Model comparison completed successfully!")
    return results


def create_comparison_plots(results: dict, x_test: np.ndarray, y_test: np.ndarray):
    """Create side-by-side comparison plots."""
    
    # 1. Training History Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy comparison
    ax1.plot(results['original']['history'].history['accuracy'], 
             label='Original (Training)', linewidth=2, color='blue')
    ax1.plot(results['original']['history'].history['val_accuracy'], 
             label='Original (Validation)', linewidth=2, color='lightblue', linestyle='--')
    ax1.plot(results['enhanced']['history'].history['accuracy'], 
             label='Enhanced (Training)', linewidth=2, color='red')
    ax1.plot(results['enhanced']['history'].history['val_accuracy'], 
             label='Enhanced (Validation)', linewidth=2, color='lightcoral', linestyle='--')
    ax1.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss comparison
    ax2.plot(results['original']['history'].history['loss'], 
             label='Original (Training)', linewidth=2, color='blue')
    ax2.plot(results['original']['history'].history['val_loss'], 
             label='Original (Validation)', linewidth=2, color='lightblue', linestyle='--')
    ax2.plot(results['enhanced']['history'].history['loss'], 
             label='Enhanced (Training)', linewidth=2, color='red')
    ax2.plot(results['enhanced']['history'].history['val_loss'], 
             label='Enhanced (Validation)', linewidth=2, color='lightcoral', linestyle='--')
    ax2.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Final metrics comparison
    metrics = ['Test Accuracy', 'Test Loss', 'Parameters (K)', 'Train Time (s)']
    original_values = [
        results['original']['test_accuracy'],
        results['original']['test_loss'],
        results['original']['parameters'] / 1000,
        results['original']['train_time']
    ]
    enhanced_values = [
        results['enhanced']['test_accuracy'],
        results['enhanced']['test_loss'],
        results['enhanced']['parameters'] / 1000,
        results['enhanced']['train_time']
    ]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x_pos - width/2, original_values, width, label='Original (2-layer)', color='blue', alpha=0.7)
    ax3.bar(x_pos + width/2, enhanced_values, width, label='Enhanced (9-layer)', color='red', alpha=0.7)
    ax3.set_title('Model Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (orig, enh) in enumerate(zip(original_values, enhanced_values)):
        ax3.text(i - width/2, orig + max(original_values + enhanced_values) * 0.01, 
                f'{orig:.3f}', ha='center', va='bottom', fontweight='bold')
        ax3.text(i + width/2, enh + max(original_values + enhanced_values) * 0.01, 
                f'{enh:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Architecture comparison
    ax4.axis('off')
    ax4.text(0.5, 0.9, 'Architecture Comparison', ha='center', va='top', 
             fontsize=16, fontweight='bold', transform=ax4.transAxes)
    
    comparison_text = f"""
Original CNN (2 Layers):
• Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
• Dense(64) → Dropout → Dense(10)
• Parameters: {results['original']['parameters']:,}
• Test Accuracy: {results['original']['test_accuracy']:.4f}

Enhanced CNN (9 Layers):
• Conv2D(32) → Conv2D(32) → MaxPool → Dropout
• Conv2D(64) → Conv2D(64) → MaxPool → Dropout  
• Conv2D(128) → MaxPool → Dropout
• Dense(512) → Dropout → Dense(10)
• Parameters: {results['enhanced']['parameters']:,}
• Test Accuracy: {results['enhanced']['test_accuracy']:.4f}

Improvement: +{(results['enhanced']['test_accuracy'] - results['original']['test_accuracy'])*100:.2f}%
"""
    
    ax4.text(0.05, 0.85, comparison_text, ha='left', va='top', 
             fontsize=10, fontfamily='monospace', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Save comparison plot
    os.makedirs("results/plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/plots/model_comparison_{timestamp}.png", 
                dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to results/plots/model_comparison_{timestamp}.png")
    plt.show()


def save_comparison_report(results: dict, epochs: int, batch_size: int):
    """Save a detailed comparison report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
CIFAR-10 CNN Model Comparison Report
Generated: {timestamp}
{'=' * 70}

TRAINING CONFIGURATION:
- Epochs: {epochs}
- Batch Size: {batch_size}
- Dataset: CIFAR-10 (50,000 train + 10,000 test)
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

ORIGINAL 2-LAYER CNN RESULTS:
{'=' * 40}
Architecture: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(64) → Dense(10)
Parameters: {results['original']['parameters']:,}
Training Time: {results['original']['train_time']:.2f} seconds

Performance:
- Test Accuracy: {results['original']['test_accuracy']:.6f} ({results['original']['test_accuracy']*100:.2f}%)
- Test Loss: {results['original']['test_loss']:.6f}
- Final Training Accuracy: {results['original']['history'].history['accuracy'][-1]:.6f}
- Final Validation Accuracy: {results['original']['history'].history['val_accuracy'][-1]:.6f}

ENHANCED 9-LAYER CNN RESULTS:
{'=' * 40}
Architecture: 3 Conv Blocks + 2 Dense Layers (with Dropout regularization)
Parameters: {results['enhanced']['parameters']:,}
Training Time: {results['enhanced']['train_time']:.2f} seconds

Performance:
- Test Accuracy: {results['enhanced']['test_accuracy']:.6f} ({results['enhanced']['test_accuracy']*100:.2f}%)
- Test Loss: {results['enhanced']['test_loss']:.6f}
- Final Training Accuracy: {results['enhanced']['history'].history['accuracy'][-1]:.6f}
- Final Validation Accuracy: {results['enhanced']['history'].history['val_accuracy'][-1]:.6f}

COMPARISON ANALYSIS:
{'=' * 40}
Accuracy Improvement: {(results['enhanced']['test_accuracy'] - results['original']['test_accuracy'])*100:+.2f} percentage points
Parameter Increase: {results['enhanced']['parameters'] / results['original']['parameters']:.1f}x
Training Time Increase: {results['enhanced']['train_time'] / results['original']['train_time']:.1f}x

CONCLUSIONS:
- The enhanced 9-layer CNN shows {'IMPROVED' if results['enhanced']['test_accuracy'] > results['original']['test_accuracy'] else 'DECREASED'} accuracy
- Deeper architecture allows for more complex feature learning
- Additional regularization (dropout) helps prevent overfitting
- Trade-off between accuracy and computational cost
"""
    
    # Save report
    os.makedirs("results", exist_ok=True)
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"results/model_comparison_report_{timestamp_file}.txt"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Comparison report saved to: {report_path}")


def quick_comparison_demo():
    """Run a quick 3-epoch comparison demo."""
    print("🚀 Running Quick Model Comparison Demo (3 epochs)")
    return train_model_comparison(epochs=3, batch_size=64, save_models=False)


def full_comparison():
    """Run full comparison with recommended settings."""
    print("🚀 Running Full Model Comparison (50 epochs)")
    return train_model_comparison(epochs=50, batch_size=32, save_models=True)


if __name__ == "__main__":
    import sys
    
    # Command line options
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            quick_comparison_demo()
        elif sys.argv[1] == "full":
            full_comparison()
        elif sys.argv[1] == "compare_arch":
            from enhanced_cnn import compare_architectures
            compare_architectures()
        else:
            print("Usage: python enhanced_train.py [demo|full|compare_arch]")
            print("  demo: Quick 3-epoch comparison")
            print("  full: Full 50-epoch comparison")
            print("  compare_arch: Just show architecture comparison")
    else:
        # Default: run demo
        print("No arguments provided. Running quick demo...")
        print("Use 'python enhanced_train.py full' for complete 50-epoch comparison")
        quick_comparison_demo()