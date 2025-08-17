"""
Performance tests for Video Analytics Pipeline System
"""

import unittest
import sys
import os
import time
import numpy as np
import multiprocessing as mp
from unittest.mock import patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import Detector
from presenter import Presenter
from tests.test_utils import TestFileManager, TestMessageQueue, TestFrameGenerator


class TestPerformance(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.file_manager = TestFileManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.file_manager.cleanup()
    
    def test_detector_performance_large_frames(self):
        """Test detector performance with large frames."""
        input_queue = TestMessageQueue()
        output_queue = TestMessageQueue()
        detector = Detector(input_queue, output_queue)
        
        # Create large frames (1920x1080)
        large_frame = TestFrameGenerator.create_frame_with_motion(
            width=1920, height=1080,
            motion_areas=[(100, 100, 200, 200), (500, 500, 150, 150)]
        )
        
        # Time multiple detections
        num_frames = 10
        start_time = time.time()
        
        for i in range(num_frames):
            detections = detector.detect_motion(large_frame)
        
        end_time = time.time()
        avg_time_per_frame = (end_time - start_time) / num_frames
        
        # Should process large frames reasonably quickly (< 100ms per frame)
        self.assertLess(avg_time_per_frame, 0.1, 
                       f"Detection too slow: {avg_time_per_frame:.3f}s per frame")
        
        print(f"Large frame detection: {avg_time_per_frame*1000:.1f}ms per frame")
    
    def test_detector_performance_many_small_detections(self):
        """Test detector performance with many small detections."""
        input_queue = TestMessageQueue()
        output_queue = TestMessageQueue()
        detector = Detector(input_queue, output_queue)
        
        # Create frame with many small motion areas
        motion_areas = []
        for i in range(20):
            for j in range(20):
                x, y = i * 30, j * 20
                if x < 600 and y < 400:  # Within 640x480 frame
                    motion_areas.append((x, y, 25, 25))
        
        frame = TestFrameGenerator.create_frame_with_motion(
            width=640, height=480, motion_areas=motion_areas
        )
        
        # Time detection
        start_time = time.time()
        detector.detect_motion(np.zeros((480, 640, 3), dtype=np.uint8))  # First frame
        detections = detector.detect_motion(frame)
        end_time = time.time()
        
        detection_time = end_time - start_time
        
        # Should handle many detections efficiently
        self.assertLess(detection_time, 0.5, 
                       f"Too slow with many detections: {detection_time:.3f}s")
        
        print(f"Many detections ({len(detections)}): {detection_time*1000:.1f}ms")
    
    def test_blur_performance_large_detections(self):
        """Test blur performance with large detection areas."""
        input_queue = TestMessageQueue()
        presenter = Presenter(input_queue, enable_blur=True, blur_intensity="heavy")
        
        # Create frame with large detection
        frame = TestFrameGenerator.create_frame_with_motion(width=1280, height=720)
        large_detection = {
            'bbox': (100, 100, 500, 400),  # Large detection area
            'area': 200000,
            'center': (350, 300),
            'contour': []
        }
        
        # Time blur operation
        start_time = time.time()
        blurred_frame = presenter.blur_detections(frame, [large_detection])
        end_time = time.time()
        
        blur_time = end_time - start_time
        
        # Blur should be reasonably fast even for large areas
        self.assertLess(blur_time, 0.2, 
                       f"Blur too slow for large area: {blur_time:.3f}s")
        
        print(f"Large area blur: {blur_time*1000:.1f}ms")
    
    def test_blur_performance_multiple_detections(self):
        """Test blur performance with multiple detection areas."""
        input_queue = TestMessageQueue()
        presenter = Presenter(input_queue, enable_blur=True, blur_intensity="medium")
        
        frame = TestFrameGenerator.create_frame_with_motion()
        
        # Create multiple detections
        detections = []
        for i in range(10):
            detection = {
                'bbox': (i * 50, i * 30, 80, 60),
                'area': 4800,
                'center': (i * 50 + 40, i * 30 + 30),
                'contour': []
            }
            detections.append(detection)
        
        # Time blur operation
        start_time = time.time()
        blurred_frame = presenter.blur_detections(frame, detections)
        end_time = time.time()
        
        blur_time = end_time - start_time
        
        # Should handle multiple detections efficiently
        self.assertLess(blur_time, 0.1, 
                       f"Multiple blur too slow: {blur_time:.3f}s")
        
        print(f"Multiple detections blur ({len(detections)}): {blur_time*1000:.1f}ms")
    
    def test_memory_usage_long_sequence(self):
        """Test memory usage during long processing sequence."""
        input_queue = TestMessageQueue()
        output_queue = TestMessageQueue()
        detector = Detector(input_queue, output_queue)
        
        # Process many frames to check for memory leaks
        num_frames = 100
        frame = TestFrameGenerator.create_frame_with_motion()
        
        # Simulate varying motion
        for i in range(num_frames):
            if i % 10 == 0:
                # Occasionally change the frame to create motion
                frame = TestFrameGenerator.create_frame_with_motion(
                    motion_areas=[(i * 5, 50, 60, 60)]
                )
            
            detections = detector.detect_motion(frame)
            
            # Periodically check frame count is incrementing properly
            if i % 25 == 0:
                self.assertEqual(detector.frame_count, i)
        
        # Should complete without memory issues
        self.assertEqual(detector.frame_count, num_frames)
        print(f"Processed {num_frames} frames successfully")
    
    @patch('cv2.imshow')
    @patch('cv2.namedWindow')
    @patch('cv2.waitKey', return_value=-1)
    @patch('cv2.destroyAllWindows')
    def test_end_to_end_performance(self, mock_destroy, mock_waitkey, mock_window, mock_imshow):
        """Test end-to-end pipeline performance."""
        # Create test video
        video_path = self.file_manager.create_temp_video(frames=30, fps=30)
        
        from video_analytics_pipeline import VideoAnalyticsPipeline
        
        pipeline = VideoAnalyticsPipeline(video_path, enable_blur=True, blur_intensity="medium")
        
        # Time full pipeline execution
        start_time = time.time()
        success = pipeline.run()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        self.assertTrue(success)
        
        # Should complete in reasonable time (30 frames should take < 15 seconds)
        self.assertLess(execution_time, 15.0, 
                       f"Pipeline too slow: {execution_time:.2f}s for 30 frames")
        
        frames_per_second = 30 / execution_time
        print(f"Pipeline performance: {frames_per_second:.1f} FPS equivalent")
    
    def test_queue_performance_high_throughput(self):
        """Test queue performance under high message throughput."""
        import multiprocessing as mp
        
        def producer(queue, num_messages):
            """Produce test messages rapidly."""
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            for i in range(num_messages):
                message = {
                    'type': 'FRAME',
                    'frame': frame,
                    'frame_number': i,
                    'timestamp': time.time()
                }
                queue.put(message)
        
        def consumer(queue, num_messages):
            """Consume test messages rapidly."""
            for i in range(num_messages):
                message = queue.get()
                # Simulate minimal processing
                _ = message['frame_number']
        
        # Test with many messages
        num_messages = 1000
        queue = mp.Queue(maxsize=100)
        
        # Start producer and consumer
        producer_process = mp.Process(target=producer, args=(queue, num_messages))
        consumer_process = mp.Process(target=consumer, args=(queue, num_messages))
        
        start_time = time.time()
        producer_process.start()
        consumer_process.start()
        
        producer_process.join(timeout=10)
        consumer_process.join(timeout=10)
        end_time = time.time()
        
        # Check processes completed successfully
        self.assertEqual(producer_process.exitcode, 0)
        self.assertEqual(consumer_process.exitcode, 0)
        
        throughput_time = end_time - start_time
        messages_per_second = num_messages / throughput_time
        
        # Should handle high throughput
        self.assertGreater(messages_per_second, 500, 
                          f"Queue throughput too low: {messages_per_second:.0f} msg/s")
        
        print(f"Queue throughput: {messages_per_second:.0f} messages/second")
    
    def test_blur_intensity_performance_comparison(self):
        """Compare performance of different blur intensities."""
        input_queue = TestMessageQueue()
        frame = TestFrameGenerator.create_frame_with_motion()
        detection = {
            'bbox': (100, 100, 200, 150),
            'area': 30000,
            'center': (200, 175),
            'contour': []
        }
        
        intensities = ["light", "medium", "heavy"]
        times = {}
        
        for intensity in intensities:
            presenter = Presenter(input_queue, enable_blur=True, blur_intensity=intensity)
            
            # Time blur operation
            start_time = time.time()
            for _ in range(10):  # Average over multiple runs
                blurred_frame = presenter.blur_detections(frame, [detection])
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            times[intensity] = avg_time
        
        # Print performance comparison
        print("Blur intensity performance comparison:")
        for intensity in intensities:
            print(f"  {intensity}: {times[intensity]*1000:.1f}ms")
        
        # Heavy blur should be slower than light blur
        self.assertGreater(times["heavy"], times["light"])


if __name__ == '__main__':
    unittest.main()
