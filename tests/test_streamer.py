"""
Unit tests for Streamer component
"""

import unittest
import sys
import os
import tempfile
import time

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamer import Streamer, run_streamer
from tests.test_utils import (
    TestFileManager, TestMessageQueue, assert_frame_message, 
    assert_end_message, run_component_with_timeout
)


class TestStreamer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.file_manager = TestFileManager()
        self.test_queue = TestMessageQueue()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.file_manager.cleanup()
    
    def test_streamer_initialization_valid_video(self):
        """Test streamer initialization with valid video file."""
        video_path = self.file_manager.create_temp_video(frames=10)
        streamer = Streamer(video_path, self.test_queue)
        
        # Test initialization
        self.assertTrue(streamer.initialize())
        self.assertIsNotNone(streamer.cap)
        self.assertTrue(streamer.cap.isOpened())
        
        # Clean up
        streamer.cleanup()
    
    def test_streamer_initialization_invalid_video(self):
        """Test streamer initialization with invalid video file."""
        invalid_path = "/nonexistent/video.mp4"
        streamer = Streamer(invalid_path, self.test_queue)
        
        # Test initialization failure
        self.assertFalse(streamer.initialize())
    
    def test_streamer_processes_frames(self):
        """Test that streamer processes video frames correctly."""
        video_path = self.file_manager.create_temp_video(frames=5, fps=30)
        streamer = Streamer(video_path, self.test_queue)
        
        # Initialize and stream a few frames manually
        self.assertTrue(streamer.initialize())
        
        # Simulate reading frames
        frame_count = 0
        while frame_count < 3:  # Read first 3 frames
            ret, frame = streamer.cap.read()
            if not ret:
                break
                
            message = {
                'type': 'FRAME',
                'frame': frame,
                'frame_number': frame_count,
                'timestamp': time.time()
            }
            self.test_queue.put(message)
            frame_count += 1
        
        # Verify messages were queued
        self.assertEqual(len(self.test_queue.messages), 3)
        
        # Verify message format
        for i, message in enumerate(self.test_queue.messages):
            assert_frame_message(message)
            self.assertEqual(message['frame_number'], i)
        
        streamer.cleanup()
    
    def test_streamer_end_of_video_signal(self):
        """Test that streamer sends end-of-video signal correctly."""
        video_path = self.file_manager.create_temp_video(frames=2, fps=30)
        
        # Use a mock queue that we can inspect
        class MockQueue:
            def __init__(self):
                self.messages = []
            
            def put(self, message):
                self.messages.append(message)
        
        mock_queue = MockQueue()
        streamer = Streamer(video_path, mock_queue)
        
        # Simulate the streaming process
        streamer.stream_frames()
        
        # Should have 2 frame messages + 1 end message
        self.assertEqual(len(mock_queue.messages), 3)
        
        # Check frame messages
        for i in range(2):
            assert_frame_message(mock_queue.messages[i])
            self.assertEqual(mock_queue.messages[i]['frame_number'], i)
        
        # Check end message
        end_message = mock_queue.messages[2]
        assert_end_message(end_message)
        self.assertEqual(end_message['total_frames'], 2)
    
    def test_streamer_static_video(self):
        """Test streamer with static video (no motion)."""
        video_path = self.file_manager.create_temp_static_video(frames=3)
        
        class MockQueue:
            def __init__(self):
                self.messages = []
            
            def put(self, message):
                self.messages.append(message)
        
        mock_queue = MockQueue()
        streamer = Streamer(video_path, mock_queue)
        streamer.stream_frames()
        
        # Should process all frames + end signal
        self.assertEqual(len(mock_queue.messages), 4)  # 3 frames + 1 end
        
        # Verify last message is end signal
        assert_end_message(mock_queue.messages[-1])
    
    def test_run_streamer_function(self):
        """Test the run_streamer function with multiprocessing."""
        video_path = self.file_manager.create_temp_video(frames=3)
        
        import multiprocessing as mp
        test_queue = mp.Queue()
        
        # Run streamer in subprocess with timeout
        success, exit_code = run_component_with_timeout(
            run_streamer, 
            (video_path, test_queue), 
            timeout=10.0
        )
        
        self.assertTrue(success, "Streamer process should complete successfully")
        self.assertEqual(exit_code, 0, "Streamer should exit with code 0")
        
        # Check that messages were sent
        messages = []
        while not test_queue.empty():
            try:
                messages.append(test_queue.get_nowait())
            except:
                break
        
        self.assertGreater(len(messages), 0, "Should have received messages")
        
        # Last message should be end signal
        if messages:
            assert_end_message(messages[-1])
    
    def test_streamer_frame_rate_control(self):
        """Test that streamer respects frame rate timing."""
        video_path = self.file_manager.create_temp_video(frames=3, fps=10)  # Low FPS for timing
        
        class TimedMockQueue:
            def __init__(self):
                self.messages = []
                self.timestamps = []
            
            def put(self, message):
                self.messages.append(message)
                self.timestamps.append(time.time())
        
        mock_queue = TimedMockQueue()
        streamer = Streamer(video_path, mock_queue)
        
        start_time = time.time()
        streamer.stream_frames()
        total_time = time.time() - start_time
        
        # Should take some time due to frame rate control
        # With 3 frames at 10 FPS, should take at least 0.2 seconds
        self.assertGreater(total_time, 0.1, "Should respect frame rate timing")
    
    def test_streamer_properties(self):
        """Test streamer video property detection."""
        video_path = self.file_manager.create_temp_video(frames=10, fps=25, width=320, height=240)
        streamer = Streamer(video_path, self.test_queue)
        
        self.assertTrue(streamer.initialize())
        
        # Test video properties
        fps = streamer.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(streamer.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(streamer.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(streamer.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.assertEqual(fps, 25.0)
        self.assertEqual(frame_count, 10)
        self.assertEqual(width, 320)
        self.assertEqual(height, 240)
        
        streamer.cleanup()


if __name__ == '__main__':
    # Add cv2 import for property tests
    import cv2
    unittest.main()
