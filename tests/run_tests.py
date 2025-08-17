#!/usr/bin/env python3
"""
Test runner for Video Analytics Pipeline System

This script runs all tests and provides a comprehensive test report.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests import test_streamer, test_detector, test_presenter, test_integration, test_utils


class TestResult:
    """Custom test result class for detailed reporting."""
    
    def __init__(self):
        self.tests_run = 0
        self.failures = []
        self.errors = []
        self.skipped = []
        self.successes = []
        self.start_time = None
        self.end_time = None
    
    def start_test(self, test):
        if self.start_time is None:
            self.start_time = time.time()
    
    def add_success(self, test):
        self.tests_run += 1
        self.successes.append(test)
    
    def add_error(self, test, err):
        self.tests_run += 1
        self.errors.append((test, err))
    
    def add_failure(self, test, err):
        self.tests_run += 1
        self.failures.append((test, err))
    
    def add_skip(self, test, reason):
        self.tests_run += 1
        self.skipped.append((test, reason))
    
    def stop_test(self, test):
        pass
    
    def stop_test_run(self):
        self.end_time = time.time()
    
    @property
    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    @property
    def success_count(self):
        return len(self.successes)
    
    @property
    def error_count(self):
        return len(self.errors)
    
    @property
    def failure_count(self):
        return len(self.failures)
    
    @property
    def skip_count(self):
        return len(self.skipped)
    
    def was_successful(self):
        return self.error_count == 0 and self.failure_count == 0


def run_test_suite(test_module, module_name):
    """Run tests from a specific module."""
    print(f"\n{'='*60}")
    print(f"Running {module_name} tests")
    print('='*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_module)
    
    # Run tests with custom result
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    # Print results
    output = stream.getvalue()
    print(output)
    
    return result


def run_all_tests():
    """Run all test suites."""
    print("Video Analytics Pipeline System - Test Suite")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Test start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test modules to run
    test_modules = [
        (test_utils, "Utilities"),
        (test_streamer, "Streamer Component"),
        (test_detector, "Detector Component"), 
        (test_presenter, "Presenter Component"),
        (test_integration, "Integration")
    ]
    
    # Track overall results
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_successes = 0
    start_time = time.time()
    
    results = []
    
    # Run each test module
    for test_module, module_name in test_modules:
        try:
            result = run_test_suite(test_module, module_name)
            results.append((module_name, result))
            
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_skipped += len(result.skipped)
            total_successes += result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
            
        except Exception as e:
            print(f"ERROR: Failed to run {module_name} tests: {e}")
            total_errors += 1
    
    # Calculate total time
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    # Print module results
    for module_name, result in results:
        status = "PASS" if result.wasSuccessful() else "FAIL"
        print(f"{module_name:.<40} {status}")
        if not result.wasSuccessful():
            if result.failures:
                print(f"  Failures: {len(result.failures)}")
            if result.errors:
                print(f"  Errors: {len(result.errors)}")
    
    print("-" * 60)
    print(f"Total Tests Run: {total_tests}")
    print(f"Successes: {total_successes}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Skipped: {total_skipped}")
    print(f"Duration: {total_duration:.2f} seconds")
    
    # Overall result
    overall_success = total_failures == 0 and total_errors == 0
    status = "PASS" if overall_success else "FAIL"
    print(f"\nOVERALL RESULT: {status}")
    
    if not overall_success:
        print("\nDETAILED FAILURE/ERROR INFORMATION:")
        print("-" * 60)
        
        for module_name, result in results:
            if result.failures or result.errors:
                print(f"\n{module_name}:")
                
                for test, traceback in result.failures:
                    print(f"  FAILURE: {test}")
                    print(f"    {traceback.strip()}")
                
                for test, traceback in result.errors:
                    print(f"  ERROR: {test}")
                    print(f"    {traceback.strip()}")
    
    return overall_success


def run_specific_test(test_pattern):
    """Run tests matching a specific pattern."""
    print(f"Running tests matching pattern: {test_pattern}")
    
    # Create test suite with pattern
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern=test_pattern)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Video Analytics Pipeline tests")
    parser.add_argument(
        "--pattern", "-p",
        help="Run only tests matching this pattern (e.g., 'test_streamer.py')",
        default=None
    )
    parser.add_argument(
        "--module", "-m",
        help="Run only specific module (streamer, detector, presenter, integration, utils)",
        choices=["streamer", "detector", "presenter", "integration", "utils"],
        default=None
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.pattern:
        success = run_specific_test(args.pattern)
    elif args.module:
        # Run specific module
        module_map = {
            "streamer": test_streamer,
            "detector": test_detector,
            "presenter": test_presenter,
            "integration": test_integration,
            "utils": test_utils
        }
        module = module_map[args.module]
        result = run_test_suite(module, args.module.title())
        success = result.wasSuccessful()
    else:
        # Run all tests
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
