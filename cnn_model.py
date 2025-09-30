"""
2-Layer CNN for CIFAR-10 Classification

This module implements a simple 2-layer Convolutional Neural Network
for classifying images from the CIFAR-10 dataset.

CIFAR-10 contains 60,000 32x32 color images in 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class SimpleCNN:
    """
    A simple 2-layer CNN for CIFAR-10 classification.
    
    Architecture:
    - Input: 32x32x3 RGB images
    - Conv Layer 1: 32 filters, 3x3 kernel, ReLU activation
    - MaxPool: 2x2
    - Conv Layer 2: 64 filters, 3x3 kernel, ReLU activation  
    - MaxPool: 2x2
    - Flatten
    - Dense: 64 units, ReLU activation
    - Dropout: 0.5
    - Output: 10 units (softmax for 10 classes)
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (32, 32, 3), num_classes: int = 10):
        """
        Initialize the CNN model.
        
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
        Build the 2-layer CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # First Convolutional Layer
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Layer
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
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
        Train the CNN model.
        
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


def create_simple_cnn() -> SimpleCNN:
    """
    Factory function to create and return a SimpleCNN instance.
    
    Returns:
        SimpleCNN instance with model built and ready for training
    """
    cnn = SimpleCNN()
    cnn.build_model()
    return cnn


if __name__ == "__main__":
    # Example usage
    print("Creating 2-Layer CNN for CIFAR-10...")
    cnn = create_simple_cnn()
    cnn.get_model_summary()