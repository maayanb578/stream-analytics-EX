import cv2
import multiprocessing as mp
import time
from typing import Optional


class Streamer:
    """
    Streamer component that reads video frames and sends them to the detector.
    """
    
    def __init__(self, video_path: str, output_queue: mp.Queue):
        self.video_path = video_path
        self.output_queue = output_queue
        self.cap = None
    
    def initialize(self) -> bool:
        """Initialize video capture."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return False
        
        print(f"Streamer: Successfully opened video {self.video_path}")
        
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video FPS: {fps}, Total frames: {frame_count}")
        
        return True
    
    def stream_frames(self):
        """Main streaming loop that reads and sends frames."""
        if not self.initialize():
            return
        
        frame_number = 0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30  # Default to 30 FPS if can't get FPS
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Streamer: End of video or failed to read frame")
                    break
                
                # Create message with frame and metadata
                message = {
                    'frame': frame,
                    'frame_number': frame_number,
                    'timestamp': time.time()
                }
                
                # Send frame to detector
                self.output_queue.put(message)
                frame_number += 1
                
                if frame_number % 100 == 0:
                    print(f"Streamer: Processed {frame_number} frames")
                
                # Control frame rate
                time.sleep(frame_delay)
                
        except KeyboardInterrupt:
            print("Streamer: Interrupted by user")
        finally:
            # Send end signal
            self.output_queue.put(None)
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        print("Streamer: Cleanup completed")


def run_streamer(video_path: str, output_queue: mp.Queue):
    """Function to run streamer in a separate process."""
    streamer = Streamer(video_path, output_queue)
    streamer.stream_frames()


if __name__ == "__main__":
    # Test the streamer component standalone
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python streamer.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    test_queue = mp.Queue()
    
    streamer = Streamer(video_path, test_queue)
    streamer.stream_frames()
