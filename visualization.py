"""
Visualization Utilities for CIFAR-10 CNN

This module provides comprehensive visualization functions for analyzing
the CNN model performance, training progress, and predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Optional
import os
from datetime import datetime


def plot_training_history(history, save_plot: bool = True) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Keras training history object
        save_plot: Whether to save the plot to file
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("results/plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"results/plots/training_history_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved to results/plots/training_history_{timestamp}.png")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         class_names: List[str],
                         save_plot: bool = True) -> None:
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True class labels (integer format)
        y_pred: Predicted class labels (integer format)
        class_names: List of class names
        save_plot: Whether to save the plot to file
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Confusion Matrix - CIFAR-10 CNN', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("results/plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"results/plots/confusion_matrix_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to results/plots/confusion_matrix_{timestamp}.png")
    
    plt.show()


def plot_sample_predictions(x_test: np.ndarray,
                           y_test: np.ndarray,
                           y_pred: np.ndarray,
                           num_samples: int = 10,
                           save_plot: bool = True) -> None:
    """
    Plot sample images with true and predicted labels.
    
    Args:
        x_test: Test images
        y_test: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        num_samples: Number of samples to display
        save_plot: Whether to save the plot to file
    """
    from data_utils import CIFAR10_CLASSES
    
    # Convert to class indices
    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_pred_probs = np.max(y_pred, axis=1)
    
    # Select samples to display (mix of correct and incorrect predictions)
    correct_indices = np.where(y_true_classes == y_pred_classes)[0]
    incorrect_indices = np.where(y_true_classes != y_pred_classes)[0]
    
    # Try to get a mix of correct and incorrect predictions
    num_correct = min(num_samples // 2, len(correct_indices))
    num_incorrect = min(num_samples - num_correct, len(incorrect_indices))
    
    selected_indices = []
    if num_correct > 0:
        selected_indices.extend(np.random.choice(correct_indices, num_correct, replace=False))
    if num_incorrect > 0:
        selected_indices.extend(np.random.choice(incorrect_indices, num_incorrect, replace=False))
    
    # Add more samples if needed
    while len(selected_indices) < num_samples:
        remaining = num_samples - len(selected_indices)
        all_indices = np.arange(len(x_test))
        available = np.setdiff1d(all_indices, selected_indices)
        additional = np.random.choice(available, min(remaining, len(available)), replace=False)
        selected_indices.extend(additional)
    
    selected_indices = selected_indices[:num_samples]
    
    # Create plot
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    fig.suptitle('Sample Predictions - CIFAR-10 CNN', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(selected_indices):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        # Get prediction info
        true_class = y_true_classes[idx]
        pred_class = y_pred_classes[idx]
        confidence = y_pred_probs[idx]
        
        # Display image
        axes[row, col].imshow(x_test[idx])
        
        # Create title with prediction info
        true_name = CIFAR10_CLASSES[true_class]
        pred_name = CIFAR10_CLASSES[pred_class]
        
        if true_class == pred_class:
            # Correct prediction - green title
            title = f"✓ {pred_name}\n({confidence:.2f})"
            color = 'green'
        else:
            # Incorrect prediction - red title
            title = f"✗ {pred_name}\n(True: {true_name})\n({confidence:.2f})"
            color = 'red'
        
        axes[row, col].set_title(title, fontsize=10, color=color, fontweight='bold')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(selected_indices), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("results/plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"results/plots/sample_predictions_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"📊 Sample predictions saved to results/plots/sample_predictions_{timestamp}.png")
    
    plt.show()


def plot_class_accuracy_comparison(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  class_names: List[str],
                                  save_plot: bool = True) -> None:
    """
    Plot per-class accuracy comparison.
    
    Args:
        y_true: True class labels (integer format)
        y_pred: Predicted class labels (integer format) 
        class_names: List of class names
        save_plot: Whether to save the plot to file
    """
    # Calculate per-class accuracy
    accuracies = []
    for i in range(len(class_names)):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])
            accuracies.append(class_accuracy)
        else:
            accuracies.append(0.0)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Per-Class Accuracy - CIFAR-10 CNN', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add average line
    avg_accuracy = np.mean(accuracies)
    plt.axhline(y=avg_accuracy, color='red', linestyle='--', linewidth=2, 
                label=f'Average: {avg_accuracy:.3f}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("results/plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"results/plots/class_accuracy_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"📊 Class accuracy plot saved to results/plots/class_accuracy_{timestamp}.png")
    
    plt.show()


def create_model_architecture_diagram(save_plot: bool = True) -> None:
    """
    Create a visual diagram of the CNN architecture.
    
    Args:
        save_plot: Whether to save the plot to file
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, '2-Layer CNN Architecture for CIFAR-10', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Input
    ax.add_patch(plt.Rectangle((0.5, 2), 1, 2, facecolor='lightblue', edgecolor='black'))
    ax.text(1, 3, 'Input\n32×32×3', ha='center', va='center', fontweight='bold')
    
    # Conv1 + MaxPool
    ax.add_patch(plt.Rectangle((2, 2), 1.5, 2, facecolor='lightgreen', edgecolor='black'))
    ax.text(2.75, 3, 'Conv2D\n32 filters\n3×3 kernel\n+\nMaxPool2D\n2×2', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Conv2 + MaxPool
    ax.add_patch(plt.Rectangle((4, 2), 1.5, 2, facecolor='lightgreen', edgecolor='black'))
    ax.text(4.75, 3, 'Conv2D\n64 filters\n3×3 kernel\n+\nMaxPool2D\n2×2', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Flatten
    ax.add_patch(plt.Rectangle((6, 2.5), 0.8, 1, facecolor='lightyellow', edgecolor='black'))
    ax.text(6.4, 3, 'Flatten', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Dense + Dropout
    ax.add_patch(plt.Rectangle((7.2, 2), 1.3, 2, facecolor='lightcoral', edgecolor='black'))
    ax.text(7.85, 3, 'Dense\n64 units\nReLU\n+\nDropout\n0.5', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Output
    ax.add_patch(plt.Rectangle((8.8, 2.25), 1, 1.5, facecolor='lightpink', edgecolor='black'))
    ax.text(9.3, 3, 'Output\n10 classes\nSoftmax', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(2, 3), xytext=(1.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(4, 3), xytext=(3.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(6, 3), xytext=(5.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(7.2, 3), xytext=(6.8, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(8.8, 3), xytext=(8.5, 3), arrowprops=arrow_props)
    
    # Add dimensions
    ax.text(1, 1.5, '32×32×3', ha='center', va='center', fontsize=8, style='italic')
    ax.text(2.75, 1.5, '15×15×32', ha='center', va='center', fontsize=8, style='italic')
    ax.text(4.75, 1.5, '6×6×64', ha='center', va='center', fontsize=8, style='italic')
    ax.text(6.4, 1.5, '2304', ha='center', va='center', fontsize=8, style='italic')
    ax.text(7.85, 1.5, '64', ha='center', va='center', fontsize=8, style='italic')
    ax.text(9.3, 1.5, '10', ha='center', va='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("results/plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"results/plots/architecture_diagram_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"📊 Architecture diagram saved to results/plots/architecture_diagram_{timestamp}.png")
    
    plt.show()


if __name__ == "__main__":
    # Create architecture diagram
    print("Creating CNN architecture diagram...")
    create_model_architecture_diagram()