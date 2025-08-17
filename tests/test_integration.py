"""
Integration tests for Video Analytics Pipeline System
"""

import unittest
import sys
import os
import time
import multiprocessing as mp
from unittest.mock import patch

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_analytics_pipeline import VideoAnalyticsPipeline
from streamer import run_streamer
from detector import run_detector
from presenter import run_presenter
from tests.test_utils import TestFileManager, run_component_with_timeout


class TestVideoAnalyticsPipelineIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.file_manager = TestFileManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.file_manager.cleanup()
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with valid video."""
        video_path = self.file_manager.create_temp_video(frames=5)
        pipeline = VideoAnalyticsPipeline(video_path)
        
        # Test basic initialization
        self.assertEqual(pipeline.video_path, video_path)
        self.assertTrue(pipeline.enable_blur)
        self.assertEqual(pipeline.blur_intensity, "medium")
        self.assertFalse(pipeline.running)
        self.assertEqual(len(pipeline.processes), 0)
    
    def test_pipeline_initialization_with_blur_options(self):
        """Test pipeline initialization with different blur settings."""
        video_path = self.file_manager.create_temp_video(frames=5)
        
        # Test with blur disabled
        pipeline1 = VideoAnalyticsPipeline(video_path, enable_blur=False)
        self.assertFalse(pipeline1.enable_blur)
        
        # Test with heavy blur
        pipeline2 = VideoAnalyticsPipeline(video_path, enable_blur=True, blur_intensity="heavy")
        self.assertTrue(pipeline2.enable_blur)
        self.assertEqual(pipeline2.blur_intensity, "heavy")
    
    def test_video_file_validation_valid(self):
        """Test video file validation with valid file."""
        video_path = self.file_manager.create_temp_video(frames=3)
        pipeline = VideoAnalyticsPipeline(video_path)
        
        self.assertTrue(pipeline.validate_video_file())
    
    def test_video_file_validation_invalid(self):
        """Test video file validation with invalid file."""
        invalid_path = "/nonexistent/video.mp4"
        pipeline = VideoAnalyticsPipeline(invalid_path)
        
        self.assertFalse(pipeline.validate_video_file())
    
    def test_queue_setup(self):
        """Test queue setup."""
        video_path = self.file_manager.create_temp_video(frames=3)
        pipeline = VideoAnalyticsPipeline(video_path)
        
        pipeline.setup_queues()
        
        # Check queues were created
        self.assertIn('streamer_to_detector', pipeline.queues)
        self.assertIn('detector_to_presenter', pipeline.queues)
        
        # Check queues are proper multiprocessing queues
        self.assertIsInstance(pipeline.queues['streamer_to_detector'], mp.Queue)
        self.assertIsInstance(pipeline.queues['detector_to_presenter'], mp.Queue)
    
    @patch('cv2.imshow')
    @patch('cv2.namedWindow') 
    @patch('cv2.waitKey', return_value=-1)
    @patch('cv2.destroyAllWindows')
    def test_component_startup_and_cleanup(self, mock_destroy, mock_waitkey, mock_window, mock_imshow):
        """Test that all components start and clean up properly."""
        video_path = self.file_manager.create_temp_video(frames=3)
        pipeline = VideoAnalyticsPipeline(video_path)
        
        # Setup
        pipeline.setup_queues()
        
        # Start components
        success = pipeline.start_components()
        self.assertTrue(success)
        self.assertEqual(len(pipeline.processes), 3)
        self.assertTrue(pipeline.running)
        
        # Verify all processes started
        process_names = [p.name for p in pipeline.processes]
        self.assertIn("Streamer", process_names)
        self.assertIn("Detector", process_names)
        self.assertIn("Presenter", process_names)
        
        # Give processes a moment to start
        time.sleep(0.1)
        
        # All should be alive initially
        for process in pipeline.processes:
            self.assertTrue(process.is_alive())
        
        # Cleanup
        pipeline.cleanup()
    
    @patch('cv2.imshow')
    @patch('cv2.namedWindow')
    @patch('cv2.waitKey', return_value=-1)
    @patch('cv2.destroyAllWindows')
    def test_full_pipeline_execution_short_video(self, mock_destroy, mock_waitkey, mock_window, mock_imshow):
        """Test complete pipeline execution with short video."""
        video_path = self.file_manager.create_temp_video(frames=5, fps=30)
        pipeline = VideoAnalyticsPipeline(video_path)
        
        # Run pipeline with timeout
        start_time = time.time()
        
        # Use timeout to prevent hanging
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Pipeline execution timed out")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        try:
            success = pipeline.run()
            self.assertTrue(success)
        except TimeoutError:
            self.fail("Pipeline execution timed out")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        execution_time = time.time() - start_time
        self.assertLess(execution_time, 25)  # Should complete well under timeout
    
    def test_component_communication_streamer_to_detector(self):
        """Test communication between Streamer and Detector."""
        video_path = self.file_manager.create_temp_video(frames=3)
        
        # Create queues
        streamer_to_detector = mp.Queue()
        detector_to_presenter = mp.Queue()
        
        # Start streamer process
        streamer_process = mp.Process(
            target=run_streamer,
            args=(video_path, streamer_to_detector)
        )
        streamer_process.start()
        
        # Start detector process  
        detector_process = mp.Process(
            target=run_detector,
            args=(streamer_to_detector, detector_to_presenter)
        )
        detector_process.start()
        
        # Wait for processes to complete
        streamer_process.join(timeout=10)
        detector_process.join(timeout=10)
        
        # Check exit codes
        self.assertEqual(streamer_process.exitcode, 0)
        self.assertEqual(detector_process.exitcode, 0)
        
        # Check that detector produced output
        messages = []
        while not detector_to_presenter.empty():
            try:
                messages.append(detector_to_presenter.get_nowait())
            except:
                break
        
        self.assertGreater(len(messages), 0)
        
        # Last message should be end signal
        last_message = messages[-1]
        self.assertEqual(last_message['type'], 'END_OF_VIDEO')
    
    @patch('cv2.imshow')
    @patch('cv2.namedWindow')
    @patch('cv2.waitKey', return_value=-1)
    @patch('cv2.destroyAllWindows')
    def test_end_to_end_message_flow(self, mock_destroy, mock_waitkey, mock_window, mock_imshow):
        """Test complete message flow from streamer to presenter."""
        video_path = self.file_manager.create_temp_video(frames=2)
        
        # Create all queues
        streamer_to_detector = mp.Queue()
        detector_to_presenter = mp.Queue()
        
        # Mock presenter that just collects messages
        def mock_presenter_run(input_queue):
            messages = []
            try:
                while True:
                    message = input_queue.get(timeout=5)
                    messages.append(message)
                    if message.get('type') == 'END_OF_VIDEO':
                        break
            except:
                pass
            return len(messages)
        
        # Start all processes
        streamer_process = mp.Process(target=run_streamer, args=(video_path, streamer_to_detector))
        detector_process = mp.Process(target=run_detector, args=(streamer_to_detector, detector_to_presenter))
        presenter_process = mp.Process(target=mock_presenter_run, args=(detector_to_presenter,))
        
        processes = [streamer_process, detector_process, presenter_process]
        
        for p in processes:
            p.start()
        
        # Wait for completion
        for p in processes:
            p.join(timeout=15)
        
        # Check all processes completed successfully
        for p in processes:
            self.assertEqual(p.exitcode, 0, f"Process {p.name} failed with exit code {p.exitcode}")
    
    def test_pipeline_with_static_video(self):
        """Test pipeline with static video (no motion)."""
        video_path = self.file_manager.create_temp_static_video(frames=3)
        
        with patch('cv2.imshow'), patch('cv2.namedWindow'), \
             patch('cv2.waitKey', return_value=-1), patch('cv2.destroyAllWindows'):
            
            pipeline = VideoAnalyticsPipeline(video_path)
            success = pipeline.run()
            self.assertTrue(success)
    
    def test_pipeline_auto_shutdown_feature(self):
        """Test that pipeline automatically shuts down when video ends."""
        video_path = self.file_manager.create_temp_video(frames=3)
        
        with patch('cv2.imshow'), patch('cv2.namedWindow'), \
             patch('cv2.waitKey', return_value=-1), patch('cv2.destroyAllWindows'):
            
            pipeline = VideoAnalyticsPipeline(video_path)
            
            start_time = time.time()
            success = pipeline.run()
            end_time = time.time()
            
            # Should complete successfully and automatically
            self.assertTrue(success)
            
            # Should not hang indefinitely
            self.assertLess(end_time - start_time, 20)  # Should finish in reasonable time
    
    def test_pipeline_blur_configurations(self):
        """Test pipeline with different blur configurations."""
        video_path = self.file_manager.create_temp_video(frames=2)
        
        blur_configs = [
            (False, "medium"),
            (True, "light"),
            (True, "medium"), 
            (True, "heavy")
        ]
        
        with patch('cv2.imshow'), patch('cv2.namedWindow'), \
             patch('cv2.waitKey', return_value=-1), patch('cv2.destroyAllWindows'):
            
            for enable_blur, intensity in blur_configs:
                with self.subTest(enable_blur=enable_blur, intensity=intensity):
                    pipeline = VideoAnalyticsPipeline(video_path, enable_blur, intensity)
                    success = pipeline.run()
                    self.assertTrue(success, f"Failed with blur={enable_blur}, intensity={intensity}")


class TestComponentInteraction(unittest.TestCase):
    """Test individual component interactions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.file_manager = TestFileManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.file_manager.cleanup()
    
    def test_streamer_detector_message_format_compatibility(self):
        """Test that Streamer output format matches Detector input expectations."""
        video_path = self.file_manager.create_temp_video(frames=2)
        
        # Test message passing
        queue = mp.Queue()
        
        # Run streamer
        streamer_process = mp.Process(target=run_streamer, args=(video_path, queue))
        streamer_process.start()
        streamer_process.join(timeout=10)
        
        # Collect messages
        messages = []
        while not queue.empty():
            messages.append(queue.get_nowait())
        
        # Verify message format compatibility
        for message in messages:
            self.assertIsInstance(message, dict)
            self.assertIn('type', message)
            
            if message['type'] == 'FRAME':
                self.assertIn('frame', message)
                self.assertIn('frame_number', message)
                self.assertIn('timestamp', message)
            elif message['type'] == 'END_OF_VIDEO':
                self.assertIn('total_frames', message)
                self.assertIn('timestamp', message)
    
    @patch('cv2.imshow')
    @patch('cv2.namedWindow')
    @patch('cv2.waitKey', return_value=-1)
    @patch('cv2.destroyAllWindows')
    def test_detector_presenter_message_format_compatibility(self, mock_destroy, mock_waitkey, mock_window, mock_imshow):
        """Test that Detector output format matches Presenter input expectations."""
        # Create test messages as Detector would produce them
        test_messages = [
            {
                'type': 'PROCESSED_FRAME',
                'frame': np.zeros((100, 100, 3), dtype=np.uint8),
                'frame_number': 0,
                'timestamp': time.time(),
                'detections': [],
                'detection_count': 0
            },
            {
                'type': 'END_OF_VIDEO',
                'total_frames': 1,
                'timestamp': time.time()
            }
        ]
        
        # Test with mock presenter
        input_queue = mp.Queue()
        for msg in test_messages:
            input_queue.put(msg)
        
        # Run presenter process
        presenter_process = mp.Process(target=run_presenter, args=(input_queue,))
        presenter_process.start()
        presenter_process.join(timeout=10)
        
        # Should complete without errors
        self.assertEqual(presenter_process.exitcode, 0)


if __name__ == '__main__':
    import numpy as np
    unittest.main()
