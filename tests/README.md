# Video Analytics Pipeline Test Suite

This directory contains comprehensive tests for the Video Analytics Pipeline System.

## Test Structure

### Test Files

- `test_streamer.py` - Unit tests for the Streamer component
- `test_detector.py` - Unit tests for the Detector component  
- `test_presenter.py` - Unit tests for the Presenter component
- `test_integration.py` - Integration tests for the complete pipeline
- `test_performance.py` - Performance benchmarks and stress tests
- `test_utils.py` - Testing utilities and helper functions
- `run_tests.py` - Main test runner script

### Test Categories

**Unit Tests:**
- Component initialization and configuration
- Message processing and format validation
- Motion detection algorithms
- Blur algorithm functionality
- Error handling and edge cases

**Integration Tests:**
- End-to-end pipeline execution
- Component communication and message flow
- Auto-shutdown functionality
- Different video types and configurations

**Performance Tests:**
- Large frame processing speed
- Blur algorithm performance
- Memory usage over long sequences
- Queue throughput benchmarks

## Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Specific Test Module
```bash
python tests/run_tests.py --module streamer
python tests/run_tests.py --module detector
python tests/run_tests.py --module presenter
python tests/run_tests.py --module integration
python tests/run_tests.py --module utils
```

### Run Specific Test Pattern
```bash
python tests/run_tests.py --pattern "test_streamer.py"
python tests/run_tests.py --pattern "*performance*"
```

### Run Individual Test Files
```bash
cd tests
python test_streamer.py
python test_detector.py
python test_presenter.py
python test_integration.py
python test_performance.py
```

## Test Requirements

The tests automatically handle dependencies and create temporary test files as needed.

### Required Libraries
- unittest (standard library)
- opencv-python (cv2)
- numpy
- multiprocessing (standard library)

### Test Video Files
Tests automatically create temporary video files for testing:
- Motion videos with moving objects
- Static videos for edge case testing
- Various resolutions and frame counts

## Test Features

### Automatic Test Video Generation
- Creates realistic test videos with moving objects
- Generates static videos for edge case testing
- Configurable resolution, frame count, and FPS

### Mock Components
- TestMessageQueue for simulating inter-process communication
- TestFrameGenerator for creating test frames with controlled motion
- TestFileManager for automatic cleanup of temporary files

### Performance Benchmarking
- Frame processing speed measurements
- Blur algorithm performance comparison
- Memory usage tracking
- Queue throughput testing

### Comprehensive Coverage
- Component initialization and cleanup
- Message format validation
- Error condition handling
- Auto-shutdown functionality testing
- Different blur configurations

## Example Test Output

```
Video Analytics Pipeline System - Test Suite
============================================================
Python version: 3.x.x
Test start time: 2023-12-07 10:30:45

============================================================
Running Utilities tests
============================================================
test_create_test_video (test_utils.TestMockVideoCreator) ... ok
test_message_queue (test_utils.TestTestMessageQueue) ... ok

============================================================
Running Streamer Component tests
============================================================
test_streamer_initialization_valid_video (test_streamer.TestStreamer) ... ok
test_streamer_processes_frames (test_streamer.TestStreamer) ... ok
test_streamer_end_of_video_signal (test_streamer.TestStreamer) ... ok

============================================================
TEST SUMMARY
============================================================
Utilities................................ PASS
Streamer Component........................... PASS
Detector Component........................... PASS
Presenter Component.......................... PASS
Integration.................................. PASS

Total Tests Run: 45
Successes: 45
Failures: 0
Errors: 0
Skipped: 0
Duration: 12.34 seconds

OVERALL RESULT: PASS
```

## Troubleshooting

### Common Issues

**OpenCV GUI Issues:**
- Tests mock OpenCV display functions to avoid GUI dependencies
- If running in headless environment, ensure DISPLAY is not set

**Multiprocessing Issues:**
- Tests use timeouts to prevent hanging
- If tests timeout, check system resources and reduce test complexity

**Video File Creation:**
- Tests create temporary video files that are automatically cleaned up
- Ensure sufficient disk space for temporary files

### Test Debugging

Enable verbose output:
```bash
python tests/run_tests.py --verbose
```

Run specific failing test:
```bash
python -m unittest tests.test_streamer.TestStreamer.test_specific_method
```

## Contributing

When adding new features:
1. Add corresponding unit tests
2. Update integration tests if needed
3. Add performance tests for computationally intensive features
4. Run full test suite before submitting changes
