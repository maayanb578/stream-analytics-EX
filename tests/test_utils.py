"""
Test utilities for Video Analytics Pipeline System
"""

import cv2
import numpy as np
import tempfile
import os
import multiprocessing as mp
from typing import List, Tuple
import time


class MockVideoCreator:
    """Utility class to create mock video files for testing."""
    
    @staticmethod
    def create_test_video(filename: str, width: int = 640, height: int = 480, 
                         frames: int = 30, fps: int = 30) -> str:
        """Create a test video file with moving objects for motion detection testing."""
        
        # Define video codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for i in range(frames):
            # Create a frame with a moving rectangle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some background noise
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            frame = frame + noise
            
            # Add a moving rectangle (simulates motion)
            rect_size = 50
            x = int((width - rect_size) * (i / frames))
            y = height // 2 - rect_size // 2
            
            cv2.rectangle(frame, (x, y), (x + rect_size, y + rect_size), (255, 255, 255), -1)
            
            # Add some random moving dots
            for j in range(3):
                dot_x = int((width - 20) * ((i + j * 10) % frames) / frames)
                dot_y = int(height * 0.25 * (j + 1))
                cv2.circle(frame, (dot_x, dot_y), 10, (100, 200, 100), -1)
            
            out.write(frame)
        
        out.release()
        return filename
    
    @staticmethod
    def create_static_video(filename: str, width: int = 640, height: int = 480,
                           frames: int = 30, fps: int = 30) -> str:
        """Create a static test video (no motion) for testing edge cases."""
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        # Create a static frame
        static_frame = np.full((height, width, 3), 128, dtype=np.uint8)
        cv2.putText(static_frame, "STATIC TEST VIDEO", (50, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        for i in range(frames):
            out.write(static_frame)
        
        out.release()
        return filename


class TestMessageQueue:
    """Mock queue for testing message passing."""
    
    def __init__(self):
        self.messages = []
        self.get_count = 0
    
    def put(self, message):
        self.messages.append(message)
    
    def get(self):
        if self.get_count < len(self.messages):
            message = self.messages[self.get_count]
            self.get_count += 1
            return message
        return None
    
    def empty(self):
        return self.get_count >= len(self.messages)
    
    def qsize(self):
        return len(self.messages) - self.get_count
    
    def get_nowait(self):
        return self.get()


class TestFrameGenerator:
    """Generate test frames for component testing."""
    
    @staticmethod
    def create_frame_with_motion(width: int = 640, height: int = 480, 
                                motion_areas: List[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Create a frame with specified motion areas."""
        frame = np.random.randint(0, 100, (height, width, 3), dtype=np.uint8)
        
        if motion_areas:
            for x, y, w, h in motion_areas:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
        
        return frame
    
    @staticmethod
    def create_test_message(frame: np.ndarray, frame_number: int = 0, 
                           msg_type: str = "FRAME") -> dict:
        """Create a test message in the expected format."""
        return {
            'type': msg_type,
            'frame': frame,
            'frame_number': frame_number,
            'timestamp': time.time()
        }


class TestFileManager:
    """Manage temporary test files."""
    
    def __init__(self):
        self.temp_files = []
    
    def create_temp_video(self, prefix: str = "test_video", **kwargs) -> str:
        """Create a temporary test video file."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', prefix=prefix, delete=False)
        temp_file.close()
        
        filename = temp_file.name
        MockVideoCreator.create_test_video(filename, **kwargs)
        self.temp_files.append(filename)
        return filename
    
    def create_temp_static_video(self, prefix: str = "static_video", **kwargs) -> str:
        """Create a temporary static test video file."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', prefix=prefix, delete=False)
        temp_file.close()
        
        filename = temp_file.name
        MockVideoCreator.create_static_video(filename, **kwargs)
        self.temp_files.append(filename)
        return filename
    
    def cleanup(self):
        """Clean up all temporary files."""
        for filename in self.temp_files:
            try:
                if os.path.exists(filename):
                    os.unlink(filename)
            except:
                pass
        self.temp_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def run_component_with_timeout(target_func, args, timeout: float = 5.0):
    """Run a component function with a timeout to prevent hanging tests."""
    process = mp.Process(target=target_func, args=args)
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return False, "Timeout"
    
    return True, process.exitcode


def assert_message_type(message: dict, expected_type: str):
    """Assert that a message has the expected type."""
    assert isinstance(message, dict), f"Expected dict message, got {type(message)}"
    assert 'type' in message, "Message missing 'type' field"
    assert message['type'] == expected_type, f"Expected {expected_type}, got {message['type']}"


def assert_frame_message(message: dict):
    """Assert that a message is a valid frame message."""
    assert_message_type(message, 'FRAME')
    assert 'frame' in message, "Frame message missing 'frame' field"
    assert 'frame_number' in message, "Frame message missing 'frame_number' field"
    assert 'timestamp' in message, "Frame message missing 'timestamp' field"
    assert isinstance(message['frame'], np.ndarray), "Frame should be numpy array"


def assert_processed_frame_message(message: dict):
    """Assert that a message is a valid processed frame message."""
    assert_message_type(message, 'PROCESSED_FRAME')
    assert 'frame' in message, "Processed frame message missing 'frame' field"
    assert 'detections' in message, "Processed frame message missing 'detections' field"
    assert 'detection_count' in message, "Processed frame message missing 'detection_count' field"
    assert isinstance(message['detections'], list), "Detections should be a list"


def assert_end_message(message: dict):
    """Assert that a message is a valid end message."""
    assert_message_type(message, 'END_OF_VIDEO')
    assert 'total_frames' in message, "End message missing 'total_frames' field"
    assert 'timestamp' in message, "End message missing 'timestamp' field"
