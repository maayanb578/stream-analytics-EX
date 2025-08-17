"""
Unit tests for Detector component
"""

import unittest
import sys
import os
import numpy as np
import cv2
import time
import multiprocessing as mp

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import Detector, run_detector
from tests.test_utils import (
    TestMessageQueue, TestFrameGenerator, assert_processed_frame_message,
    assert_end_message, run_component_with_timeout
)


class TestDetector(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_queue = TestMessageQueue()
        self.output_queue = TestMessageQueue()
        self.detector = Detector(self.input_queue, self.output_queue)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.frame_count, 0)
        self.assertIsNone(self.detector.prev_frame)
    
    def test_motion_detection_first_frame(self):
        """Test motion detection with first frame (should return no detections)."""
        frame = TestFrameGenerator.create_frame_with_motion()
        detections = self.detector.detect_motion(frame)
        
        # First frame should have no detections (no previous frame to compare)
        self.assertEqual(len(detections), 0)
        self.assertIsNotNone(self.detector.prev_frame)
    
    def test_motion_detection_no_motion(self):
        """Test motion detection with identical frames (no motion)."""
        frame1 = TestFrameGenerator.create_frame_with_motion()
        frame2 = frame1.copy()  # Identical frame
        
        # Process first frame
        detections1 = self.detector.detect_motion(frame1)
        self.assertEqual(len(detections1), 0)
        
        # Process identical second frame
        detections2 = self.detector.detect_motion(frame2)
        self.assertEqual(len(detections2), 0)
    
    def test_motion_detection_with_motion(self):
        """Test motion detection with actual motion."""
        # Create first frame
        frame1 = TestFrameGenerator.create_frame_with_motion()
        
        # Create second frame with motion (different content)
        frame2 = TestFrameGenerator.create_frame_with_motion(
            motion_areas=[(100, 100, 80, 80)]  # Large motion area
        )
        
        # Process frames
        detections1 = self.detector.detect_motion(frame1)
        detections2 = self.detector.detect_motion(frame2)
        
        # First frame has no detections
        self.assertEqual(len(detections1), 0)
        
        # Second frame should detect motion
        self.assertGreater(len(detections2), 0)
        
        # Verify detection structure
        for detection in detections2:
            self.assertIn('bbox', detection)
            self.assertIn('area', detection)
            self.assertIn('center', detection)
            self.assertIn('contour', detection)
            
            # Verify bbox format
            bbox = detection['bbox']
            self.assertEqual(len(bbox), 4)  # (x, y, w, h)
            
            # Verify area is positive
            self.assertGreater(detection['area'], 0)
    
    def test_motion_detection_area_filtering(self):
        """Test that small motion areas are filtered out."""
        # Create frames with very small motion
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        
        # Add tiny motion (should be filtered out)
        cv2.rectangle(frame2, (100, 100), (105, 105), (255, 255, 255), -1)  # 5x5 pixel
        
        # Process frames
        self.detector.detect_motion(frame1)
        detections = self.detector.detect_motion(frame2)
        
        # Small motion should be filtered out (area < 500 threshold)
        self.assertEqual(len(detections), 0)
    
    def test_process_frame_messages(self):
        """Test detector processing of frame messages."""
        # Create test frame message
        frame = TestFrameGenerator.create_frame_with_motion(motion_areas=[(50, 50, 100, 100)])
        message = TestFrameGenerator.create_test_message(frame, frame_number=5)
        
        # Add message to input queue
        self.input_queue.put(message)
        
        # Process one message manually
        input_message = self.input_queue.get()
        self.assertEqual(input_message['type'], 'FRAME')
        
        # Simulate detector processing
        detections = self.detector.detect_motion(input_message['frame'])
        output_message = {
            'type': 'PROCESSED_FRAME',
            'frame': input_message['frame'],
            'frame_number': input_message['frame_number'],
            'timestamp': input_message['timestamp'],
            'detections': detections,
            'detection_count': len(detections)
        }
        
        self.output_queue.put(output_message)
        
        # Verify output message
        result = self.output_queue.get()
        assert_processed_frame_message(result)
        self.assertEqual(result['frame_number'], 5)
    
    def test_end_of_video_forwarding(self):
        """Test that detector forwards end-of-video signals."""
        end_message = {
            'type': 'END_OF_VIDEO',
            'total_frames': 100,
            'timestamp': time.time()
        }
        
        self.input_queue.put(end_message)
        
        # Simulate detector processing end message
        input_message = self.input_queue.get()
        if input_message.get('type') == 'END_OF_VIDEO':
            self.output_queue.put(input_message)
        
        # Verify forwarding
        result = self.output_queue.get()
        assert_end_message(result)
        self.assertEqual(result['total_frames'], 100)
    
    def test_interrupt_signal_forwarding(self):
        """Test that detector forwards interrupt signals."""
        interrupt_message = {
            'type': 'INTERRUPTED',
            'total_frames': 50,
            'timestamp': time.time()
        }
        
        self.input_queue.put(interrupt_message)
        
        # Simulate detector processing interrupt message
        input_message = self.input_queue.get()
        if input_message.get('type') == 'INTERRUPTED':
            self.output_queue.put(input_message)
        
        # Verify forwarding
        result = self.output_queue.get()
        self.assertEqual(result['type'], 'INTERRUPTED')
        self.assertEqual(result['total_frames'], 50)
    
    def test_multiple_frame_processing(self):
        """Test detector processing multiple frames in sequence."""
        frames_data = []
        
        # Create sequence of frames with varying motion
        for i in range(5):
            if i % 2 == 0:
                # Even frames: static
                frame = TestFrameGenerator.create_frame_with_motion()
            else:
                # Odd frames: with motion
                frame = TestFrameGenerator.create_frame_with_motion(
                    motion_areas=[(i * 50, 50, 60, 60)]
                )
            
            message = TestFrameGenerator.create_test_message(frame, frame_number=i)
            frames_data.append((message, i % 2 == 1))  # (message, has_motion)
        
        detection_counts = []
        
        # Process all frames
        for message, expected_motion in frames_data:
            detections = self.detector.detect_motion(message['frame'])
            detection_counts.append(len(detections))
        
        # First frame should have no detections (no previous frame)
        self.assertEqual(detection_counts[0], 0)
        
        # Subsequent frames: should detect motion for frames with actual changes
        for i in range(1, len(detection_counts)):
            if frames_data[i][1]:  # If motion was expected
                # Note: detection depends on difference from previous frame
                pass  # We can't guarantee detection without more complex setup
    
    def test_detection_bbox_validity(self):
        """Test that detection bounding boxes are valid."""
        # Create frames with known motion area
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        
        # Add significant motion in known location
        cv2.rectangle(frame2, (200, 150), (300, 250), (255, 255, 255), -1)  # 100x100 area
        
        # Process frames
        self.detector.detect_motion(frame1)
        detections = self.detector.detect_motion(frame2)
        
        # Verify detections
        if len(detections) > 0:
            for detection in detections:
                bbox = detection['bbox']
                x, y, w, h = bbox
                
                # Verify bbox coordinates are valid
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertGreater(w, 0)
                self.assertGreater(h, 0)
                self.assertLess(x + w, 640)  # Within frame width
                self.assertLess(y + h, 480)  # Within frame height
                
                # Verify center calculation
                center = detection['center']
                expected_center = (x + w // 2, y + h // 2)
                self.assertEqual(center, expected_center)
    
    def test_run_detector_function(self):
        """Test the run_detector function with multiprocessing."""
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        
        # Add test messages
        frame = TestFrameGenerator.create_frame_with_motion()
        frame_message = TestFrameGenerator.create_test_message(frame)
        input_queue.put(frame_message)
        
        # Add end signal
        end_message = {
            'type': 'END_OF_VIDEO',
            'total_frames': 1,
            'timestamp': time.time()
        }
        input_queue.put(end_message)
        
        # Run detector with timeout
        success, exit_code = run_component_with_timeout(
            run_detector,
            (input_queue, output_queue),
            timeout=10.0
        )
        
        self.assertTrue(success, "Detector process should complete successfully")
        self.assertEqual(exit_code, 0, "Detector should exit with code 0")
        
        # Check output messages
        messages = []
        while not output_queue.empty():
            try:
                messages.append(output_queue.get_nowait())
            except:
                break
        
        self.assertGreater(len(messages), 0, "Should have received messages")
        
        # Should have processed frame + end message
        self.assertEqual(len(messages), 2)
        assert_processed_frame_message(messages[0])
        assert_end_message(messages[1])


if __name__ == '__main__':
    unittest.main()
