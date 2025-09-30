"""
Enhanced 9-Layer CNN for CIFAR-10 Classification

This module implements a deeper 9-layer Convolutional Neural Network
for improved accuracy on the CIFAR-10 dataset. This allows direct
comparison with the original 2-layer model using the same data.

Architecture:
- Input: 32×32×3 RGB images
- Conv Block 1: Conv2D(32) + Conv2D(32) + MaxPool + Dropout
- Conv Block 2: Conv2D(64) + Conv2D(64) + MaxPool + Dropout  
- Conv Block 3: Conv2D(128) + MaxPool + Dropout
- Dense Block: Dense(512) + Dropout + Dense(10)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class EnhancedCNN:
    """
    A deeper 9-layer CNN for CIFAR-10 classification with improved accuracy.
    
    Architecture (9 layers total):
    - Input: 32×32×3 RGB images
    - Conv Layer 1: 32 filters, 3×3 kernel + ReLU
    - Conv Layer 2: 32 filters, 3×3 kernel + ReLU  
    - MaxPool: 2×2 + Dropout(0.25)
    - Conv Layer 3: 64 filters, 3×3 kernel + ReLU
    - Conv Layer 4: 64 filters, 3×3 kernel + ReLU
    - MaxPool: 2×2 + Dropout(0.25)
    - Conv Layer 5: 128 filters, 3×3 kernel + ReLU
    - MaxPool: 2×2 + Dropout(0.25)
    - Flatten
    - Dense Layer 1: 512 units + ReLU + Dropout(0.5)
    - Dense Layer 2: 10 units (softmax for 10 classes)
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (32, 32, 3), num_classes: int = 10):
        """
        Initialize the Enhanced CNN model.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Build the 9-layer CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First Convolutional Block (2 Conv layers)
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block (2 Conv layers)
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block (1 Conv layer)
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Block
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_model_summary(self) -> None:
        """Print model architecture summary."""
        if self.model is None:
            self.build_model()
        self.model.summary()
    
    def train(self, 
              x_train: np.ndarray, 
              y_train: np.ndarray,
              x_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 10,
              batch_size: int = 32,
              verbose: int = 1) -> keras.callbacks.History:
        """
        Train the Enhanced CNN model.
        
        Args:
            x_train: Training images
            y_train: Training labels (one-hot encoded)
            x_val: Validation images
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
            
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=verbose
        )
        
        self.history = history
        return history
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Args:
            x_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation")
            
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return test_loss, test_accuracy
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            x: Input images
            
        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
            
        return self.model.predict(x)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model must be built before saving")
        self.model.save(filepath)
        
    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model."""
        self.model = keras.models.load_model(filepath)


def create_enhanced_cnn() -> EnhancedCNN:
    """
    Factory function to create and return an EnhancedCNN instance.
    
    Returns:
        EnhancedCNN instance with model built and ready for training
    """
    cnn = EnhancedCNN()
    cnn.build_model()
    return cnn


def compare_architectures():
    """Compare the original 2-layer vs enhanced 9-layer architectures."""
    from cnn_model import SimpleCNN
    
    print("🔍 CNN Architecture Comparison")
    print("=" * 60)
    
    # Original 2-layer model
    print("\n📊 Original 2-Layer CNN:")
    print("-" * 30)
    simple_cnn = SimpleCNN()
    simple_cnn.build_model()
    simple_cnn.get_model_summary()
    
    simple_params = simple_cnn.model.count_params()
    
    print(f"\n📊 Enhanced 9-Layer CNN:")
    print("-" * 30)
    enhanced_cnn = EnhancedCNN()
    enhanced_cnn.build_model()
    enhanced_cnn.get_model_summary()
    
    enhanced_params = enhanced_cnn.model.count_params()
    
    print(f"\n📈 Comparison Summary:")
    print(f"  Original Model: {simple_params:,} parameters")
    print(f"  Enhanced Model: {enhanced_params:,} parameters")
    print(f"  Parameter Increase: {enhanced_params/simple_params:.1f}x")
    print(f"  Expected Accuracy Improvement: ~15-25% higher")


if __name__ == "__main__":
    # Compare architectures
    compare_architectures()