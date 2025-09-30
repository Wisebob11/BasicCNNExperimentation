#!/usr/bin/env python3
"""
Quick Test Runner for CIFAR-10 CNN Project

This script provides quick commands to run specific test categories.
Usage:
    python quick_test.py unit        # Run only unit tests
    python quick_test.py integration # Run only integration tests  
    python quick_test.py benchmark   # Run only performance benchmark
    python quick_test.py all         # Run all tests (default)
"""

import sys
import unittest
from test import (
    TestDataUtils, TestCNNModel, TestTrainingPipeline, 
    TestVisualization, TestIntegrationWithRealData,
    run_performance_benchmark
)


def run_unit_tests():
    """Run only unit tests (no real data loading)."""
    print("🧪 Running Unit Tests Only...")
    print("=" * 40)
    
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add only unit test classes
    unit_test_classes = [
        TestDataUtils,
        TestCNNModel,
        TestTrainingPipeline,
        TestVisualization
    ]
    
    for test_class in unit_test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0


def run_integration_tests():
    """Run only integration tests (with real CIFAR-10 data)."""
    print("🔗 Running Integration Tests Only...")
    print("=" * 40)
    print("Note: This will download CIFAR-10 data if not already cached")
    
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add only integration test class
    tests = test_loader.loadTestsFromTestCase(TestIntegrationWithRealData)
    test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0


def run_benchmark_only():
    """Run only performance benchmark."""
    print("⚡ Running Performance Benchmark Only...")
    print("=" * 40)
    
    results = run_performance_benchmark()
    return results is not None


def run_all_tests():
    """Run complete test suite."""
    print("🧪 Running Complete Test Suite...")
    print("=" * 40)
    
    from test import main
    return main()


def print_usage():
    """Print usage information."""
    print("CIFAR-10 CNN Quick Test Runner")
    print("=" * 30)
    print("Usage:")
    print("  python quick_test.py unit        # Unit tests only (fast)")
    print("  python quick_test.py integration # Integration tests only")
    print("  python quick_test.py benchmark   # Performance benchmark only")
    print("  python quick_test.py all         # Complete test suite (default)")
    print("")
    print("Examples:")
    print("  python quick_test.py unit        # Quick validation (~30 seconds)")
    print("  python quick_test.py integration # Test with real data (~2 minutes)")
    print("  python quick_test.py benchmark   # Performance test (~1 minute)")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        test_type = "all"
    else:
        test_type = sys.argv[1].lower()
    
    if test_type == "help" or test_type == "--help" or test_type == "-h":
        print_usage()
        return
    
    success = False
    
    if test_type == "unit":
        success = run_unit_tests()
    elif test_type == "integration":
        success = run_integration_tests()
    elif test_type == "benchmark":
        success = run_benchmark_only()
    elif test_type == "all":
        success = run_all_tests()
    else:
        print(f"❌ Unknown test type: {test_type}")
        print_usage()
        return
    
    # Print final result
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Some tests failed or had errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()