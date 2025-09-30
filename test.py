"""
Test Suite for CIFAR-10 CNN Project

This module contains comprehensive tests for all components of the CNN project:
1. Unit tests for data utilities
2. Model architecture tests  
3. Training pipeline tests
4. Integration tests with real CIFAR-10 data

Run with: python test.py
"""

import unittest
import numpy as np
import tensorflow as tf
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import our modules
from data_utils import (
    load_cifar10_data, get_dataset_info, get_class_distribution,
    show_sample_images, CIFAR10_CLASSES
)
from cnn_model import SimpleCNN, create_simple_cnn
from train import train_and_evaluate_cnn, save_training_summary
from visualization import (
    plot_training_history, plot_confusion_matrix, 
    create_model_architecture_diagram
)


class TestDataUtils(unittest.TestCase):
    """Test data loading and preprocessing utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cifar10_classes_constant(self):
        """Test that CIFAR10_CLASSES contains correct class names."""
        expected_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.assertEqual(CIFAR10_CLASSES, expected_classes)
        self.assertEqual(len(CIFAR10_CLASSES), 10)
    
    def test_get_dataset_info(self):
        """Test dataset information function."""
        info = get_dataset_info()
        
        # Check required keys
        required_keys = [
            'name', 'description', 'classes', 'num_classes',
            'image_shape', 'training_samples', 'test_samples'
        ]
        for key in required_keys:
            self.assertIn(key, info)
        
        # Check values
        self.assertEqual(info['name'], 'CIFAR-10')
        self.assertEqual(info['num_classes'], 10)
        self.assertEqual(info['image_shape'], (32, 32, 3))
        self.assertEqual(info['training_samples'], 50000)
        self.assertEqual(info['test_samples'], 10000)
        self.assertEqual(info['classes'], CIFAR10_CLASSES)
    
    def test_get_class_distribution_onehot(self):
        """Test class distribution with one-hot encoded labels."""
        # Create mock one-hot encoded data
        y_data = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # airplane
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # airplane
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # automobile
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # bird
        ])
        
        distribution = get_class_distribution(y_data)
        
        self.assertEqual(distribution['airplane'], 2)
        self.assertEqual(distribution['automobile'], 1)
        self.assertEqual(distribution['bird'], 1)
        # Note: Classes not present won't be in the dictionary
        self.assertNotIn('cat', distribution)
    
    def test_get_class_distribution_integer(self):
        """Test class distribution with integer labels."""
        y_data = np.array([0, 0, 1, 2, 0])  # 3 airplanes, 1 automobile, 1 bird
        
        distribution = get_class_distribution(y_data)
        
        self.assertEqual(distribution['airplane'], 3)
        self.assertEqual(distribution['automobile'], 1)
        self.assertEqual(distribution['bird'], 1)
    
    @patch('matplotlib.pyplot.show')
    def test_show_sample_images(self, mock_show):
        """Test sample image display function."""
        # Create mock image data
        x_data = np.random.rand(10, 32, 32, 3)
        y_data = np.random.randint(0, 10, 10)
        
        # Should not raise an exception
        try:
            show_sample_images(x_data, y_data, num_samples=5)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"show_sample_images raised an exception: {e}")
        
        # Check that matplotlib show was called
        mock_show.assert_called_once()


class TestCNNModel(unittest.TestCase):
    """Test CNN model implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cnn = SimpleCNN()
        
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        self.assertEqual(self.cnn.input_shape, (32, 32, 3))
        self.assertEqual(self.cnn.num_classes, 10)
        self.assertIsNone(self.cnn.model)
        self.assertIsNone(self.cnn.history)
    
    def test_model_initialization_custom(self):
        """Test model initialization with custom parameters."""
        cnn = SimpleCNN(input_shape=(64, 64, 3), num_classes=5)
        self.assertEqual(cnn.input_shape, (64, 64, 3))
        self.assertEqual(cnn.num_classes, 5)
    
    def test_build_model(self):
        """Test model building."""
        model = self.cnn.build_model()
        
        # Check that model is created
        self.assertIsNotNone(model)
        self.assertIsNotNone(self.cnn.model)
        
        # Check model structure (Conv, MaxPool, Conv, MaxPool, Flatten, Dense, Dropout, Dense)
        self.assertEqual(len(model.layers), 8)
        
        # Check input and output shapes
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))
        
        # Check that model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
    
    def test_create_simple_cnn_factory(self):
        """Test factory function for creating CNN."""
        cnn = create_simple_cnn()
        
        self.assertIsInstance(cnn, SimpleCNN)
        self.assertIsNotNone(cnn.model)
    
    def test_model_predict_before_build(self):
        """Test that prediction fails before model is built."""
        x_dummy = np.random.rand(1, 32, 32, 3)
        
        with self.assertRaises(ValueError):
            self.cnn.predict(x_dummy)
    
    def test_model_evaluate_before_build(self):
        """Test that evaluation fails before model is built."""
        x_dummy = np.random.rand(1, 32, 32, 3)
        y_dummy = np.random.rand(1, 10)
        
        with self.assertRaises(ValueError):
            self.cnn.evaluate(x_dummy, y_dummy)
    
    def test_model_save_before_build(self):
        """Test that saving fails before model is built."""
        with self.assertRaises(ValueError):
            self.cnn.save_model("test_model")


class TestTrainingPipeline(unittest.TestCase):
    """Test training pipeline functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_training_summary(self):
        """Test training summary saving."""
        # Create mock objects
        cnn = SimpleCNN()
        cnn.build_model()
        
        # Create mock history
        mock_history = MagicMock()
        mock_history.history = {
            'loss': [1.5, 1.2, 1.0],
            'accuracy': [0.4, 0.6, 0.7],
            'val_loss': [1.6, 1.3, 1.1],
            'val_accuracy': [0.35, 0.55, 0.65]
        }
        
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            save_training_summary(
                cnn, mock_history, 1.1, 0.65, 3, 32, 
                "0:01:30"
            )
            
            # Check that results directory and file were created
            self.assertTrue(os.path.exists("results"))
            result_files = os.listdir("results")
            self.assertTrue(any(f.startswith("training_summary_") for f in result_files))
            
        finally:
            os.chdir(original_dir)


class TestVisualization(unittest.TestCase):
    """Test visualization utilities."""
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_training_history(self, mock_savefig, mock_show):
        """Test training history plotting."""
        # Create mock history
        mock_history = MagicMock()
        mock_history.history = {
            'accuracy': [0.4, 0.6, 0.7],
            'val_accuracy': [0.35, 0.55, 0.65],
            'loss': [1.5, 1.2, 1.0],
            'val_loss': [1.6, 1.3, 1.1]
        }
        
        # Should not raise an exception
        try:
            plot_training_history(mock_history, save_plot=False)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_training_history raised an exception: {e}")
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_confusion_matrix(self, mock_savefig, mock_show):
        """Test confusion matrix plotting."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        class_names = ['class0', 'class1', 'class2']
        
        # Should not raise an exception
        try:
            plot_confusion_matrix(y_true, y_pred, class_names, save_plot=False)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_confusion_matrix raised an exception: {e}")
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_create_model_architecture_diagram(self, mock_savefig, mock_show):
        """Test architecture diagram creation."""
        # Should not raise an exception
        try:
            create_model_architecture_diagram(save_plot=False)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"create_model_architecture_diagram raised an exception: {e}")


class TestIntegrationWithRealData(unittest.TestCase):
    """Integration tests with real CIFAR-10 data."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures (load data once)."""
        print("\n🔄 Loading CIFAR-10 data for integration tests...")
        try:
            # Load a small subset for testing
            (cls.x_train, cls.y_train), (cls.x_val, cls.y_val), (cls.x_test, cls.y_test) = load_cifar10_data(validation_split=0.1)
            
            # Use only a small subset for faster testing
            cls.x_train_small = cls.x_train[:100]
            cls.y_train_small = cls.y_train[:100]
            cls.x_val_small = cls.x_val[:20]
            cls.y_val_small = cls.y_val[:20]
            cls.x_test_small = cls.x_test[:50]
            cls.y_test_small = cls.y_test[:50]
            
            cls.data_loaded = True
            print("✅ Test data loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load CIFAR-10 data: {e}")
            cls.data_loaded = False
    
    def setUp(self):
        """Skip tests if data couldn't be loaded."""
        if not self.data_loaded:
            self.skipTest("CIFAR-10 data not available")
    
    def test_data_shapes_and_types(self):
        """Test that loaded data has correct shapes and types."""
        # Check training data
        self.assertEqual(len(self.x_train_small.shape), 4)  # (samples, height, width, channels)
        self.assertEqual(self.x_train_small.shape[1:], (32, 32, 3))
        self.assertEqual(len(self.y_train_small.shape), 2)  # (samples, classes)
        self.assertEqual(self.y_train_small.shape[1], 10)
        
        # Check data types (TensorFlow may use float64 in some cases)
        self.assertIn(self.x_train_small.dtype, [np.float32, np.float64])
        self.assertIn(self.y_train_small.dtype, [np.float32, np.float64])
        
        # Check value ranges
        self.assertTrue(np.all(self.x_train_small >= 0))
        self.assertTrue(np.all(self.x_train_small <= 1))
        
        # Check one-hot encoding
        self.assertTrue(np.allclose(np.sum(self.y_train_small, axis=1), 1))
    
    def test_model_training_minimal(self):
        """Test that model can train on real data (1 epoch)."""
        cnn = SimpleCNN()
        cnn.build_model()
        
        # Train for just 1 epoch with small data
        history = cnn.train(
            self.x_train_small, self.y_train_small,
            self.x_val_small, self.y_val_small,
            epochs=1,
            batch_size=16,
            verbose=0
        )
        
        # Check that training produced history
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)
        self.assertIn('val_loss', history.history)
        self.assertIn('val_accuracy', history.history)
        
        # Check that we have exactly 1 epoch of data
        self.assertEqual(len(history.history['loss']), 1)
    
    def test_model_prediction(self):
        """Test model prediction on real data."""
        cnn = SimpleCNN()
        model = cnn.build_model()
        
        # Make predictions (even without training)
        predictions = cnn.predict(self.x_test_small)
        
        # Check prediction shape and properties
        self.assertEqual(predictions.shape, (len(self.x_test_small), 10))
        
        # Check that predictions are probabilities (sum to 1)
        prob_sums = np.sum(predictions, axis=1)
        self.assertTrue(np.allclose(prob_sums, 1.0, rtol=1e-5))
        
        # Check that all probabilities are between 0 and 1
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
    
    def test_model_evaluation(self):
        """Test model evaluation on real data."""
        cnn = SimpleCNN()
        cnn.build_model()
        
        # Evaluate model (even without training)
        loss, accuracy = cnn.evaluate(self.x_test_small, self.y_test_small)
        
        # Check that we get reasonable values
        self.assertIsInstance(loss, (float, np.floating))
        self.assertIsInstance(accuracy, (float, np.floating))
        self.assertTrue(loss >= 0)
        self.assertTrue(0 <= accuracy <= 1)


def run_performance_benchmark():
    """Run a performance benchmark on the CNN."""
    print("\n🏃‍♂️ Running Performance Benchmark...")
    print("=" * 50)
    
    try:
        # Load small dataset
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(validation_split=0.1)
        
        # Use subset for benchmark
        x_train_bench = x_train[:1000]
        y_train_bench = y_train[:1000] 
        x_val_bench = x_val[:200]
        y_val_bench = y_val[:200]
        x_test_bench = x_test[:500]
        y_test_bench = y_test[:500]
        
        print(f"Benchmark dataset: {len(x_train_bench)} train, {len(x_val_bench)} val, {len(x_test_bench)} test")
        
        # Create and train model
        import time
        start_time = time.time()
        
        cnn = SimpleCNN()
        cnn.build_model()
        
        build_time = time.time() - start_time
        print(f"⏱️  Model build time: {build_time:.3f} seconds")
        
        # Training benchmark
        train_start = time.time()
        history = cnn.train(
            x_train_bench, y_train_bench,
            x_val_bench, y_val_bench,
            epochs=2,
            batch_size=32,
            verbose=0
        )
        train_time = time.time() - train_start
        
        print(f"⏱️  Training time (2 epochs): {train_time:.3f} seconds")
        print(f"⏱️  Time per epoch: {train_time/2:.3f} seconds")
        
        # Evaluation benchmark
        eval_start = time.time()
        test_loss, test_accuracy = cnn.evaluate(x_test_bench, y_test_bench)
        eval_time = time.time() - eval_start
        
        print(f"⏱️  Evaluation time: {eval_time:.3f} seconds")
        
        # Performance results
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        print(f"\n📊 Performance Results:")
        print(f"  Final Training Accuracy: {final_train_acc:.4f}")
        print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        
        return {
            'build_time': build_time,
            'train_time': train_time,
            'eval_time': eval_time,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss
        }
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return None


def main():
    """Main test runner."""
    print("🧪 CIFAR-10 CNN Test Suite")
    print("=" * 60)
    
    # Run unit tests
    print("\n📋 Running Unit Tests...")
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataUtils,
        TestCNNModel, 
        TestTrainingPipeline,
        TestVisualization,
        TestIntegrationWithRealData
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    # Run performance benchmark
    benchmark_results = run_performance_benchmark()
    
    # Overall result
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n✅ All tests passed! The CIFAR-10 CNN implementation is working correctly.")
    else:
        print(f"\n❌ Some tests failed. Please review the output above.")
    
    return success


if __name__ == "__main__":
    main()