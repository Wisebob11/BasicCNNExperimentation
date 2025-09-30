# CIFAR-10 CNN Classifier 🚀

A simple yet effective 2-layer Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset. This project provides an easy-to-use implementation with comprehensive visualization and evaluation tools.

## 📋 Overview

This project implements **two CNN architectures** for direct performance comparison:

### 🔥 **Enhanced 9-Layer CNN** (Recommended)
- **Input**: 32×32×3 RGB images
- **Conv Block 1**: Conv2D(32) + Conv2D(32) + MaxPool + Dropout(0.25)
- **Conv Block 2**: Conv2D(64) + Conv2D(64) + MaxPool + Dropout(0.25)
- **Conv Block 3**: Conv2D(128) + MaxPool + Dropout(0.25)
- **Dense Block**: Dense(512) + Dropout(0.5) + Dense(10)
- **Parameters**: 1,193,642 (4.55 MB)
- **Expected Accuracy**: ~69% (3 epochs), ~80-85% (50+ epochs)

### 📊 **Original 2-Layer CNN** (Baseline)
- **Input**: 32×32×3 RGB images
- **Conv Layer 1**: 32 filters, 3×3 kernel + ReLU + MaxPool(2×2)
- **Conv Layer 2**: 64 filters, 3×3 kernel + ReLU + MaxPool(2×2)
- **Dense Layer**: 64 units + ReLU + Dropout(0.5) + Dense(10)
- **Parameters**: 167,562 (654 KB)
- **Expected Accuracy**: ~55% (3 epochs), ~70-75% (50+ epochs)

## 🎯 CIFAR-10 Dataset

The CIFAR-10 dataset consists of 60,000 32×32 color images in 10 classes:
- 🛩️ **airplane**
- 🚗 **automobile** 
- 🐦 **bird**
- 🐱 **cat**
- 🦌 **deer**
- 🐕 **dog**
- 🐸 **frog**
- 🐎 **horse**
- 🚢 **ship**
- 🚛 **truck**

**Dataset Details:**
- Training samples: 50,000
- Test samples: 10,000
- Image size: 32×32×3 (RGB)
- Download size: ~170MB
- **Automatic download**: The dataset is automatically downloaded on first use!

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd BasicCNNExperimentation

# The Python environment and packages are already configured!
# Required packages: tensorflow, numpy, matplotlib, seaborn, scikit-learn
```

### 2. Compare Model Architectures

```bash
# Compare 2-layer vs 9-layer architectures
python enhanced_train.py compare_arch
```

### 3. Run Quick Comparison Demo (3 epochs)

```bash
# Train both models and compare performance
python enhanced_train.py demo
```

### 4. Run Full Comparison (50 epochs)

```bash
# Complete training with detailed analysis
python enhanced_train.py full
```

### 5. Original Training Options

```bash
# Original 2-layer model only
python train.py demo    # 3 epochs
python train.py full    # 20 epochs
```

## 📁 Project Structure

```
BasicCNNExperimentation/
├── README.md                 # Project documentation
├── cnn_model.py             # Original 2-layer CNN implementation
├── enhanced_cnn.py          # Enhanced 9-layer CNN implementation
├── data_utils.py            # Data loading and preprocessing
├── train.py                 # Original training pipeline
├── enhanced_train.py        # Enhanced training with model comparison
├── visualization.py         # Visualization utilities
├── example.py               # Quick demo script
├── test.py                  # Comprehensive test suite
├── quick_test.py            # Quick test runner
├── requirements.txt         # Python dependencies
├── models/                  # Saved trained models
└── results/                 # Training results and plots
    ├── plots/              # Generated visualizations
    ├── model_comparison_*.png   # Side-by-side model comparisons
    ├── training_summary_*.txt   # Training session summaries
    └── model_comparison_report_*.txt  # Detailed comparison reports
```

## 💾 How to Easily Add/Load Data

### Automatic Data Loading (Recommended)

The easiest way to get CIFAR-10 data is to use our built-in utilities:

```python
from data_utils import load_cifar10_data, print_dataset_summary

# Print dataset information
print_dataset_summary()

# Load and preprocess data (automatically downloads CIFAR-10)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data()

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
```

**What happens automatically:**
1. ✅ Downloads CIFAR-10 dataset (~170MB) to `~/.keras/datasets/`
2. ✅ Normalizes pixel values to [0, 1] range
3. ✅ Converts labels to one-hot encoding
4. ✅ Splits data into train/validation/test sets
5. ✅ Returns preprocessed, ready-to-use data

### View Sample Data

```python
from data_utils import show_sample_images, get_class_distribution

# Show sample images
show_sample_images(x_train, y_train, num_samples=10)

# Check class distribution
distribution = get_class_distribution(y_train)
print(distribution)
```

### Manual Data Loading (Advanced)

If you want more control:

```python
import tensorflow as tf

# Load raw CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Manual preprocessing
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

## 🎨 Visualization Features

The project includes comprehensive visualization tools:

### Training Progress
- Loss and accuracy curves over time
- Training vs validation metrics

### Model Performance
- Confusion matrix
- Per-class accuracy comparison
- Sample predictions with confidence scores

### Architecture Diagram
- Visual representation of the CNN layers

```python
from visualization import plot_training_history, plot_confusion_matrix
from visualization import create_model_architecture_diagram

# Create architecture diagram
create_model_architecture_diagram()
```

## 📊 Model Performance

### 🔥 **Enhanced 9-Layer CNN Results** (3 epochs demo):
- **Test Accuracy**: 69.05% 
- **Training Time**: ~117 seconds
- **Parameters**: 1,193,642
- **Improvement**: +13.61% over baseline

### 📈 **Original 2-Layer CNN Results** (3 epochs demo):
- **Test Accuracy**: 55.44%
- **Training Time**: ~28 seconds  
- **Parameters**: 167,562
- **Good for**: Quick experiments and learning

### 🏆 **Why the Enhanced Model Performs Better:**
1. **Deeper Architecture**: More layers = more complex feature learning
2. **Better Regularization**: Strategic dropout placement prevents overfitting
3. **Larger Dense Layer**: 512 units vs 64 for better representation
4. **Same-padding Convolutions**: Preserves spatial information better

## 🔧 Customization Options

### Model Architecture

```python
from cnn_model import SimpleCNN

# Create custom model
cnn = SimpleCNN(input_shape=(32, 32, 3), num_classes=10)
model = cnn.build_model()

# View architecture
cnn.get_model_summary()
```

### Training Parameters

```python
# Customize training
history = cnn.train(
    x_train, y_train,
    x_val, y_val,
    epochs=20,          # Number of training epochs
    batch_size=32,      # Batch size
    verbose=1           # Verbosity level
)
```

### Data Preprocessing

```python
from data_utils import load_cifar10_data

# Custom validation split
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(
    validation_split=0.2  # Use 20% for validation
)
```

## 📈 Results and Outputs

After training, you'll find:

### Generated Files
- `models/cifar10_cnn_YYYYMMDD_HHMMSS/`: Saved trained model
- `results/training_summary_YYYYMMDD_HHMMSS.txt`: Training session summary
- `results/plots/`: Various visualization plots

### Console Output
- Dataset loading progress
- Model architecture summary
- Training progress with metrics
- Final test results
- Per-class performance report

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all packages are installed
2. **Memory Issues**: Reduce batch size if you encounter memory errors
3. **Slow Training**: Consider using GPU if available

### GPU Support

To use GPU acceleration (if available):

```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

## 🧪 Testing

The project includes a comprehensive test suite to validate all components:

### Run All Tests

```bash
python test.py
```

### Test Categories

1. **Unit Tests**: Test individual functions and components
   - Data loading utilities
   - Model architecture validation
   - Training pipeline components
   - Visualization functions

2. **Integration Tests**: Test with real CIFAR-10 data
   - Data shape and type validation
   - Model training functionality
   - Prediction and evaluation

3. **Performance Benchmark**: Measure system performance
   - Model build time
   - Training speed
   - Evaluation time
   - Memory usage

### Test Results

The test suite validates:
- ✅ Data loading and preprocessing
- ✅ Model architecture (8 layers total)
- ✅ Training pipeline functionality
- ✅ Visualization generation
- ✅ Integration with real CIFAR-10 data
- ✅ Performance benchmarks

### Test Data

The tests use both:
- **Mock data**: For fast unit tests
- **Real CIFAR-10 data**: For integration tests (automatically downloaded)
- **Subset training**: Small datasets for performance testing

## 🔬 Extending the Project

Ideas for enhancements:
- Add data augmentation
- Experiment with different architectures
- Implement learning rate scheduling
- Add more visualization options
- Try different optimizers
- Add more comprehensive test coverage

## 📚 Learning Resources

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [CNN Architecture Guide](https://cs231n.github.io/convolutional-networks/)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this educational project!

---

**Happy Learning! 🎓**