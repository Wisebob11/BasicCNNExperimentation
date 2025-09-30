"""
CIFAR-10 Data Loading and Preprocessing Utilities

This module provides easy-to-use functions for loading and preprocessing
the CIFAR-10 dataset. The CIFAR-10 dataset is automatically downloaded
by TensorFlow/Keras the first time you use it.

CIFAR-10 Dataset Information:
- 60,000 32x32 color images in 10 classes
- 50,000 training images and 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_cifar10_data(validation_split: float = 0.1) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                             Tuple[np.ndarray, np.ndarray],
                                                             Tuple[np.ndarray, np.ndarray]]:
    """
    Load and preprocess CIFAR-10 dataset.
    
    This function automatically downloads CIFAR-10 data if it's not already present.
    The data is downloaded to ~/.keras/datasets/ directory.
    
    Args:
        validation_split: Fraction of training data to use for validation (0.0 to 1.0)
        
    Returns:
        Tuple containing:
        - (x_train, y_train): Training data
        - (x_val, y_val): Validation data  
        - (x_test, y_test): Test data
        
    Example:
        # Load the data
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data()
        
        # Data shapes:
        # x_train: (45000, 32, 32, 3) - training images
        # y_train: (45000, 10) - training labels (one-hot encoded)
        # x_val: (5000, 32, 32, 3) - validation images  
        # y_val: (5000, 10) - validation labels (one-hot encoded)
        # x_test: (10000, 32, 32, 3) - test images
        # y_test: (10000, 10) - test labels (one-hot encoded)
    """
    print("Loading CIFAR-10 dataset...")
    print("Note: If this is your first time, the dataset will be downloaded (~170MB)")
    
    # Load the raw data from Keras
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    print(f"Downloaded dataset:")
    print(f"  Training images: {x_train_full.shape}")
    print(f"  Training labels: {y_train_full.shape}")
    print(f"  Test images: {x_test.shape}")
    print(f"  Test labels: {y_test.shape}")
    
    # Split training data into train and validation sets
    if validation_split > 0:
        val_size = int(len(x_train_full) * validation_split)
        
        # Shuffle the data before splitting
        indices = np.random.permutation(len(x_train_full))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        x_train = x_train_full[train_indices]
        y_train = y_train_full[train_indices]
        x_val = x_train_full[val_indices]
        y_val = y_train_full[val_indices]
    else:
        x_train = x_train_full
        y_train = y_train_full
        x_val = np.array([])
        y_val = np.array([])
    
    # Normalize pixel values to [0, 1] range
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0 if len(x_val) > 0 else x_val
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical (one-hot encoding)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10) if len(y_val) > 0 else y_val
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"\nPreprocessed data:")
    print(f"  Training: {x_train.shape} images, {y_train.shape} labels")
    if len(x_val) > 0:
        print(f"  Validation: {x_val.shape} images, {y_val.shape} labels")
    print(f"  Test: {x_test.shape} images, {y_test.shape} labels")
    print(f"  Pixel values normalized to [0, 1]")
    print(f"  Labels converted to one-hot encoding")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_dataset_info() -> dict:
    """
    Get information about the CIFAR-10 dataset.
    
    Returns:
        Dictionary containing dataset information
    """
    return {
        'name': 'CIFAR-10',
        'description': 'The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes',
        'classes': CIFAR10_CLASSES,
        'num_classes': len(CIFAR10_CLASSES),
        'image_shape': (32, 32, 3),
        'training_samples': 50000,
        'test_samples': 10000,
        'download_size': '~170MB',
        'download_location': '~/.keras/datasets/'
    }


def show_sample_images(x_data: np.ndarray, 
                      y_data: np.ndarray, 
                      num_samples: int = 10,
                      title: str = "Sample Images") -> None:
    """
    Display sample images from the dataset.
    
    Args:
        x_data: Image data (normalized or unnormalized)
        y_data: Label data (one-hot encoded or integer labels)
        num_samples: Number of images to display
        title: Title for the plot
    """
    # Convert one-hot to integer labels if needed
    if len(y_data.shape) > 1 and y_data.shape[1] > 1:
        labels = np.argmax(y_data, axis=1)
    else:
        labels = y_data.flatten()
    
    # Create subplot
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    
    for i in range(min(num_samples, 10)):
        row = i // 5
        col = i % 5
        
        # Display image
        axes[row, col].imshow(x_data[i])
        axes[row, col].set_title(f'{CIFAR10_CLASSES[labels[i]]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


def get_class_distribution(y_data: np.ndarray) -> dict:
    """
    Get the distribution of classes in the dataset.
    
    Args:
        y_data: Label data (one-hot encoded or integer labels)
        
    Returns:
        Dictionary mapping class names to counts
    """
    # Convert one-hot to integer labels if needed
    if len(y_data.shape) > 1 and y_data.shape[1] > 1:
        labels = np.argmax(y_data, axis=1)
    else:
        labels = y_data.flatten()
    
    # Count each class
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create distribution dictionary
    distribution = {}
    for class_idx, count in zip(unique, counts):
        distribution[CIFAR10_CLASSES[class_idx]] = count
    
    return distribution


def print_dataset_summary():
    """Print a comprehensive summary of the CIFAR-10 dataset."""
    info = get_dataset_info()
    
    print("=" * 50)
    print("CIFAR-10 DATASET SUMMARY")
    print("=" * 50)
    print(f"Dataset: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Image Shape: {info['image_shape']}")
    print(f"Number of Classes: {info['num_classes']}")
    print(f"Training Samples: {info['training_samples']}")
    print(f"Test Samples: {info['test_samples']}")
    print(f"Download Size: {info['download_size']}")
    print(f"Download Location: {info['download_location']}")
    
    print(f"\nClasses:")
    for i, class_name in enumerate(info['classes']):
        print(f"  {i}: {class_name}")
    
    print("\nHow to use:")
    print("1. Call load_cifar10_data() to download and preprocess the data")
    print("2. The data will be automatically downloaded on first use")
    print("3. Images are normalized to [0,1] range")
    print("4. Labels are converted to one-hot encoding")
    print("=" * 50)


if __name__ == "__main__":
    # Example usage
    print_dataset_summary()
    
    # Load and display sample data
    print("\nLoading sample data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data()
    
    # Show class distribution
    print("\nTraining set class distribution:")
    train_dist = get_class_distribution(y_train)
    for class_name, count in train_dist.items():
        print(f"  {class_name}: {count}")
    
    # Display sample images
    show_sample_images(x_train, y_train, title="CIFAR-10 Training Samples")