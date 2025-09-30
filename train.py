"""
Training and Evaluation Script for CIFAR-10 CNN

This script demonstrates how to train and evaluate the 2-layer CNN
on the CIFAR-10 dataset with comprehensive logging and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from cnn_model import SimpleCNN
from data_utils import load_cifar10_data, show_sample_images, get_class_distribution
from visualization import plot_training_history, plot_confusion_matrix, plot_sample_predictions


def train_and_evaluate_cnn(epochs: int = 10, 
                          batch_size: int = 32,
                          validation_split: float = 0.1,
                          save_model: bool = True) -> SimpleCNN:
    """
    Complete training and evaluation pipeline for CIFAR-10 CNN.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data for validation
        save_model: Whether to save the trained model
        
    Returns:
        Trained SimpleCNN instance
    """
    print("🚀 Starting CIFAR-10 CNN Training Pipeline")
    print("=" * 60)
    
    # 1. Load and preprocess data
    print("📁 Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(validation_split)
    
    # Display data information
    print(f"\nDataset loaded successfully!")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    
    # Show class distribution
    print("\nClass distribution in training set:")
    train_dist = get_class_distribution(y_train)
    for class_name, count in train_dist.items():
        print(f"  {class_name:12}: {count:5d} samples")
    
    # 2. Create and build model
    print("\n🏗️  Building CNN model...")
    cnn = SimpleCNN()
    cnn.build_model()
    cnn.get_model_summary()
    
    # 3. Train the model
    print(f"\n🎯 Training model for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split}")
    
    start_time = datetime.now()
    
    history = cnn.train(
        x_train, y_train,
        x_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    training_time = datetime.now() - start_time
    print(f"\n✅ Training completed in {training_time}")
    
    # 4. Evaluate on test set
    print("\n📊 Evaluating model on test set...")
    test_loss, test_accuracy = cnn.evaluate(x_test, y_test)
    
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 5. Generate predictions for analysis
    print("\n🔍 Generating predictions for analysis...")
    y_pred = cnn.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate per-class accuracy
    from sklearn.metrics import classification_report
    print("\nPer-class performance:")
    from data_utils import CIFAR10_CLASSES
    report = classification_report(y_true_classes, y_pred_classes, 
                                 target_names=CIFAR10_CLASSES)
    print(report)
    
    # 6. Save model if requested
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/cifar10_cnn_{timestamp}"
        os.makedirs("models", exist_ok=True)
        cnn.save_model(model_path)
        print(f"\n💾 Model saved to: {model_path}")
    
    # 7. Create visualizations
    print("\n📈 Creating visualizations...")
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, CIFAR10_CLASSES)
    
    # Show sample predictions
    plot_sample_predictions(x_test, y_test, y_pred, num_samples=10)
    
    # Save training summary
    save_training_summary(cnn, history, test_loss, test_accuracy, 
                         epochs, batch_size, training_time)
    
    print("\n🎉 Training pipeline completed successfully!")
    return cnn


def save_training_summary(cnn: SimpleCNN, 
                         history, 
                         test_loss: float, 
                         test_accuracy: float,
                         epochs: int,
                         batch_size: int,
                         training_time) -> None:
    """Save a summary of the training session."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"""
CIFAR-10 CNN Training Summary
Generated: {timestamp}
{'=' * 50}

Model Architecture:
- 2-Layer CNN for CIFAR-10 classification
- Input: 32x32x3 RGB images
- Conv Layer 1: 32 filters, 3x3 kernel + MaxPool
- Conv Layer 2: 64 filters, 3x3 kernel + MaxPool
- Dense: 64 units + Dropout(0.5)
- Output: 10 classes (softmax)

Training Configuration:
- Epochs: {epochs}
- Batch Size: {batch_size}
- Training Time: {training_time}

Final Results:
- Test Loss: {test_loss:.4f}
- Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)

Training History:
- Final Training Loss: {history.history['loss'][-1]:.4f}
- Final Training Accuracy: {history.history['accuracy'][-1]:.4f}
- Final Validation Loss: {history.history['val_loss'][-1]:.4f}
- Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}

Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}
Epoch with Best Val Acc: {np.argmax(history.history['val_accuracy']) + 1}
"""
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Save summary
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"results/training_summary_{timestamp_file}.txt"
    
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"📄 Training summary saved to: {summary_path}")


def quick_demo():
    """Run a quick demo with fewer epochs for testing."""
    print("🚀 Running Quick Demo (3 epochs)")
    cnn = train_and_evaluate_cnn(epochs=3, batch_size=64, save_model=False)
    return cnn


def full_training():
    """Run full training with recommended settings."""
    print("🚀 Running Full Training (20 epochs)")
    cnn = train_and_evaluate_cnn(epochs=20, batch_size=32, save_model=True)
    return cnn


if __name__ == "__main__":
    import sys
    
    # Command line options
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            quick_demo()
        elif sys.argv[1] == "full":
            full_training()
        else:
            print("Usage: python train.py [demo|full]")
            print("  demo: Quick 3-epoch training for testing")
            print("  full: Full 20-epoch training")
    else:
        # Default: run demo
        print("No arguments provided. Running quick demo...")
        print("Use 'python train.py full' for complete training")
        quick_demo()