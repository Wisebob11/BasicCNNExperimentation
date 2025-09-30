"""
Simple Example: CIFAR-10 CNN Demo

This script demonstrates the basic usage of the CIFAR-10 CNN classifier.
Perfect for getting started quickly!
"""

from data_utils import load_cifar10_data, show_sample_images, print_dataset_summary
from cnn_model import SimpleCNN
from visualization import create_model_architecture_diagram
import numpy as np


def main():
    """Run a simple demonstration of the CIFAR-10 CNN."""
    print("🎯 CIFAR-10 CNN Simple Demo")
    print("=" * 50)
    
    # 1. Show dataset information
    print("\n📊 Dataset Information:")
    print_dataset_summary()
    
    # 2. Load a small sample of data for demo
    print("\n📁 Loading sample data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(validation_split=0.1)
    
    # 3. Show sample images
    print("\n🖼️  Displaying sample images...")
    show_sample_images(x_train, y_train, num_samples=10, title="CIFAR-10 Sample Images")
    
    # 4. Create and show model
    print("\n🏗️  Creating CNN model...")
    cnn = SimpleCNN()
    model = cnn.build_model()
    cnn.get_model_summary()
    
    # 5. Show model architecture diagram
    print("\n📐 Creating architecture diagram...")
    create_model_architecture_diagram()
    
    # 6. Show data shapes
    print(f"\n📋 Data Shapes:")
    print(f"  Training data: {x_train.shape}")
    print(f"  Training labels: {y_train.shape}")
    print(f"  Validation data: {x_val.shape}")
    print(f"  Validation labels: {y_val.shape}")
    print(f"  Test data: {x_test.shape}")
    print(f"  Test labels: {y_test.shape}")
    
    # 7. Show next steps
    print(f"\n🚀 Next Steps:")
    print(f"  1. Run 'python train.py demo' for quick 3-epoch training")
    print(f"  2. Run 'python train.py full' for complete 20-epoch training")
    print(f"  3. Check the results/ folder for plots and summaries")
    print(f"  4. Explore the saved models in the models/ folder")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"The CIFAR-10 data is now cached at ~/.keras/datasets/ for future use.")


if __name__ == "__main__":
    main()