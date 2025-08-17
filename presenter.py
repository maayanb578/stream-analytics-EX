import cv2
import multiprocessing as mp
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Any


class Presenter:
    """
    Presenter component that draws detections and displays video frames.
    """
    
    def __init__(self, input_queue: mp.Queue):
        self.input_queue = input_queue
        self.frame_count = 0
        self.window_name = "Video Analytics System"
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection bounding boxes and information on the frame.
        """
        # Create a copy of the frame to draw on
        display_frame = frame.copy()
        
        # Draw each detection
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            area = detection['area']
            center = detection['center']
            
            x, y, w, h = bbox
            
            # Draw bounding rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(display_frame, center, 5, (0, 0, 255), -1)
            
            # Add detection info text
            info_text = f"Motion {i+1}: Area={int(area)}"
            cv2.putText(display_frame, info_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return display_frame
    
    def add_timestamp(self, frame: np.ndarray) -> np.ndarray:
        """
        Add current timestamp to the upper left corner of the frame.
        """
        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)  # White text
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(current_time, font, font_scale, thickness)
        
        # Draw black background rectangle for better text visibility
        cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + 10), (0, 0, 0), -1)
        
        # Draw timestamp text
        cv2.putText(frame, current_time, (15, 10 + text_height), font, font_scale, color, thickness)
        
        return frame
    
    def add_statistics(self, frame: np.ndarray, detection_count: int, frame_number: int) -> np.ndarray:
        """
        Add statistics information to the frame.
        """
        # Add detection count
        stats_text = f"Detections: {detection_count} | Frame: {frame_number}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 0)  # Yellow text
        thickness = 2
        
        # Position at bottom left
        height, width = frame.shape[:2]
        text_y = height - 20
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(stats_text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, text_y - text_height - 5), (10 + text_width + 10, text_y + 5), (0, 0, 0), -1)
        
        # Draw statistics text
        cv2.putText(frame, stats_text, (15, text_y), font, font_scale, color, thickness)
        
        return frame
    
    def display_frames(self):
        """Main display loop that receives processed frames and shows them."""
        print("Presenter: Starting video display")
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                # Get message from detector
                message = self.input_queue.get()
                
                # Check for end signal
                if message is None:
                    print("Presenter: Received end signal")
                    break
                
                frame = message['frame']
                detections = message['detections']
                detection_count = message['detection_count']
                frame_number = message['frame_number']
                
                # Draw detections on frame
                display_frame = self.draw_detections(frame, detections)
                
                # Add timestamp
                display_frame = self.add_timestamp(display_frame)
                
                # Add statistics
                display_frame = self.add_statistics(display_frame, detection_count, frame_number)
                
                # Display the frame
                cv2.imshow(self.window_name, display_frame)
                
                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    print(f"Presenter: Displayed {self.frame_count} frames")
                
                # Check for quit key (ESC or 'q')
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q' key
                    print("Presenter: User requested quit")
                    break
                
        except KeyboardInterrupt:
            print("Presenter: Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
        print("Presenter: Cleanup completed")


def run_presenter(input_queue: mp.Queue):
    """Function to run presenter in a separate process."""
    presenter = Presenter(input_queue)
    presenter.display_frames()


if __name__ == "__main__":
    # Test the presenter component standalone
    print("Presenter component - run as part of the pipeline system")
