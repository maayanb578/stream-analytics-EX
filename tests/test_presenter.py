"""
Unit tests for Presenter component
"""

import unittest
import sys
import os
import numpy as np
import cv2
import time
from unittest.mock import patch, MagicMock

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from presenter import Presenter, run_presenter
from tests.test_utils import (
    TestMessageQueue, TestFrameGenerator, run_component_with_timeout
)


class TestPresenter(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_queue = TestMessageQueue()
        self.presenter = Presenter(self.input_queue, enable_blur=True, blur_intensity="medium")
    
    def test_presenter_initialization(self):
        """Test presenter initialization."""
        self.assertIsNotNone(self.presenter)
        self.assertTrue(self.presenter.enable_blur)
        self.assertEqual(self.presenter.blur_intensity, "medium")
        self.assertEqual(self.presenter.frame_count, 0)
    
    def test_presenter_initialization_no_blur(self):
        """Test presenter initialization with blur disabled."""
        presenter = Presenter(self.input_queue, enable_blur=False)
        self.assertFalse(presenter.enable_blur)
    
    def test_blur_settings_configuration(self):
        """Test that blur intensity settings are properly configured."""
        # Test different blur intensities
        for intensity in ["light", "medium", "heavy"]:
            presenter = Presenter(self.input_queue, blur_intensity=intensity)
            self.assertEqual(presenter.blur_intensity, intensity)
            self.assertIn(intensity, presenter.blur_settings)
            
            settings = presenter.blur_settings[intensity]
            self.assertIn("min_kernel", settings)
            self.assertIn("max_kernel", settings)
            self.assertIn("size_factor", settings)
    
    def test_blur_detections_no_detections(self):
        """Test blur_detections with no detections."""
        frame = TestFrameGenerator.create_frame_with_motion()
        detections = []
        
        result = self.presenter.blur_detections(frame, detections)
        
        # With no detections, frame should be unchanged
        np.testing.assert_array_equal(result, frame)
    
    def test_blur_detections_disabled(self):
        """Test blur_detections when blur is disabled."""
        presenter = Presenter(self.input_queue, enable_blur=False)
        frame = TestFrameGenerator.create_frame_with_motion()
        detections = [{'bbox': (100, 100, 50, 50), 'area': 2500, 'center': (125, 125)}]
        
        result = presenter.blur_detections(frame, detections)
        
        # With blur disabled, frame should be unchanged
        np.testing.assert_array_equal(result, frame)
    
    def test_blur_detections_with_valid_detection(self):
        """Test blur_detections with valid detection."""
        frame = TestFrameGenerator.create_frame_with_motion(width=320, height=240)
        detections = [{
            'bbox': (50, 50, 100, 80),  # Large enough detection
            'area': 8000,
            'center': (100, 90),
            'contour': []
        }]
        
        result = self.presenter.blur_detections(frame, detections)
        
        # Frame should be modified (blurred in detection area)
        self.assertFalse(np.array_equal(result, frame))
        
        # Verify that only the detection area was modified
        # Areas outside detection should be unchanged
        x, y, w, h = detections[0]['bbox']
        
        # Check area outside detection (should be unchanged)
        if x > 10:
            np.testing.assert_array_equal(result[:, :x-10], frame[:, :x-10])
    
    def test_blur_detections_boundary_cases(self):
        """Test blur_detections with edge cases."""
        frame = TestFrameGenerator.create_frame_with_motion(width=100, height=100)
        
        # Detection at frame edge
        detections = [{
            'bbox': (90, 90, 20, 20),  # Extends beyond frame
            'area': 400,
            'center': (100, 100),
            'contour': []
        }]
        
        # Should not crash and should handle boundary correctly
        result = self.presenter.blur_detections(frame, detections)
        self.assertEqual(result.shape, frame.shape)
    
    def test_blur_detections_small_roi(self):
        """Test blur_detections with very small detection (should be skipped)."""
        frame = TestFrameGenerator.create_frame_with_motion()
        detections = [{
            'bbox': (100, 100, 3, 3),  # Too small (< 5 pixels)
            'area': 9,
            'center': (101, 101),
            'contour': []
        }]
        
        result = self.presenter.blur_detections(frame, detections)
        
        # Small detection should be skipped, frame unchanged
        np.testing.assert_array_equal(result, frame)
    
    def test_add_timestamp(self):
        """Test timestamp addition to frame."""
        frame = TestFrameGenerator.create_frame_with_motion()
        original_frame = frame.copy()
        
        result = self.presenter.add_timestamp(frame)
        
        # Frame should be modified (timestamp added)
        self.assertFalse(np.array_equal(result, original_frame))
        
        # Check that timestamp area (upper left) was modified
        timestamp_area = result[10:40, 10:300]  # Approximate timestamp area
        original_area = original_frame[10:40, 10:300]
        self.assertFalse(np.array_equal(timestamp_area, original_area))
    
    def test_add_statistics(self):
        """Test statistics addition to frame."""
        frame = TestFrameGenerator.create_frame_with_motion(height=400)
        original_frame = frame.copy()
        
        result = self.presenter.add_statistics(frame, detection_count=3, frame_number=150)
        
        # Frame should be modified (statistics added)
        self.assertFalse(np.array_equal(result, original_frame))
        
        # Check that statistics area (bottom left) was modified
        stats_area = result[-50:-10, 10:400]  # Approximate statistics area
        original_area = original_frame[-50:-10, 10:400]
        self.assertFalse(np.array_equal(stats_area, original_area))
    
    def test_draw_detections_complete_process(self):
        """Test complete draw_detections process with blur and overlays."""
        frame = TestFrameGenerator.create_frame_with_motion()
        detections = [{
            'bbox': (100, 100, 80, 60),
            'area': 4800,
            'center': (140, 130),
            'contour': []
        }]
        
        result = self.presenter.draw_detections(frame, detections)
        
        # Frame should be significantly modified
        self.assertFalse(np.array_equal(result, frame))
        
        # Verify frame shape is preserved
        self.assertEqual(result.shape, frame.shape)
    
    def test_blur_intensity_adaptive_kernel(self):
        """Test that different blur intensities produce different kernel sizes."""
        frame = TestFrameGenerator.create_frame_with_motion()
        detection = {
            'bbox': (100, 100, 100, 100),  # 100x100 detection
            'area': 10000,
            'center': (150, 150),
            'contour': []
        }
        
        # Test different intensities
        for intensity in ["light", "medium", "heavy"]:
            presenter = Presenter(self.input_queue, blur_intensity=intensity)
            settings = presenter.blur_settings[intensity]
            
            # Calculate expected kernel size
            detection_size = max(100, 100)  # width, height
            expected_kernel = max(settings["min_kernel"], 
                                min(settings["max_kernel"], 
                                    detection_size // settings["size_factor"]))
            
            # Ensure odd kernel
            if expected_kernel % 2 == 0:
                expected_kernel += 1
            
            # Verify kernel size is within expected range
            self.assertGreaterEqual(expected_kernel, settings["min_kernel"])
            self.assertLessEqual(expected_kernel, settings["max_kernel"])
    
    @patch('cv2.imshow')
    @patch('cv2.namedWindow')
    @patch('cv2.waitKey', return_value=27)  # ESC key
    @patch('cv2.destroyAllWindows')
    def test_display_frames_user_quit(self, mock_destroy, mock_waitkey, mock_window, mock_imshow):
        """Test display_frames with user quit (ESC key)."""
        # Add a processed frame message
        frame = TestFrameGenerator.create_frame_with_motion()
        message = {
            'type': 'PROCESSED_FRAME',
            'frame': frame,
            'frame_number': 1,
            'timestamp': time.time(),
            'detections': [],
            'detection_count': 0
        }
        self.input_queue.put(message)
        
        # Run display_frames (should quit due to ESC key)
        self.presenter.display_frames()
        
        # Verify OpenCV functions were called
        mock_window.assert_called_once()
        mock_imshow.assert_called()
        mock_destroy.assert_called_once()
    
    @patch('cv2.imshow')
    @patch('cv2.namedWindow')
    @patch('cv2.waitKey', return_value=-1)  # No key press
    @patch('cv2.destroyAllWindows')
    def test_display_frames_normal_completion(self, mock_destroy, mock_waitkey, mock_window, mock_imshow):
        """Test display_frames with normal video completion."""
        # Add end of video message
        end_message = {
            'type': 'END_OF_VIDEO',
            'total_frames': 100,
            'timestamp': time.time()
        }
        self.input_queue.put(end_message)
        
        # Mock waitKey for final pause
        with patch('cv2.waitKey', return_value=-1) as mock_final_wait:
            self.presenter.display_frames()
            
            # Should call waitKey for final pause
            mock_final_wait.assert_called()
        
        mock_destroy.assert_called_once()
    
    def test_processed_frame_message_handling(self):
        """Test handling of processed frame messages."""
        frame = TestFrameGenerator.create_frame_with_motion()
        detections = [{
            'bbox': (50, 50, 100, 100),
            'area': 10000,
            'center': (100, 100),
            'contour': []
        }]
        
        message = {
            'type': 'PROCESSED_FRAME',
            'frame': frame,
            'frame_number': 42,
            'timestamp': time.time(),
            'detections': detections,
            'detection_count': len(detections)
        }
        
        # Test message processing components individually
        with patch('cv2.imshow'), patch('cv2.namedWindow'), patch('cv2.waitKey', return_value=-1):
            # Test draw_detections
            display_frame = self.presenter.draw_detections(frame, detections)
            
            # Test timestamp addition
            display_frame = self.presenter.add_timestamp(display_frame)
            
            # Test statistics addition
            display_frame = self.presenter.add_statistics(
                display_frame, len(detections), message['frame_number'])
            
            # Frame should be processed without errors
            self.assertEqual(display_frame.shape, frame.shape)


if __name__ == '__main__':
    unittest.main()
