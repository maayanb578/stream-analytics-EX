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
    
    def __init__(self, input_queue: mp.Queue, enable_blur: bool = True, blur_intensity: str = "medium"):
        self.input_queue = input_queue
        self.frame_count = 0
        self.window_name = "Video Analytics System"
        self.enable_blur = enable_blur
        self.blur_intensity = blur_intensity
        
        # Blur intensity settings
        self.blur_settings = {
            "light": {"min_kernel": 9, "max_kernel": 21, "size_factor": 15},
            "medium": {"min_kernel": 15, "max_kernel": 35, "size_factor": 10},
            "heavy": {"min_kernel": 25, "max_kernel": 51, "size_factor": 8}
        }
    
    def blur_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Apply blur to detected motion areas in the frame.
        Uses Gaussian blur for efficient and smooth blurring.
        """
        # Skip blurring if disabled
        if not self.enable_blur or not detections:
            return frame
        
        # Create a copy of the frame to blur
        blurred_frame = frame.copy()
        
        # Get blur settings for current intensity
        settings = self.blur_settings.get(self.blur_intensity, self.blur_settings["medium"])
        
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # Ensure bounding box is within frame boundaries
            frame_height, frame_width = frame.shape[:2]
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = max(1, min(w, frame_width - x))
            h = max(1, min(h, frame_height - y))
            
            # Skip if ROI is too small
            if w < 5 or h < 5:
                continue
            
            # Extract the region of interest (ROI)
            roi = blurred_frame[y:y+h, x:x+w]
            
            # Calculate adaptive kernel size based on detection size and intensity
            kernel_size = max(settings["min_kernel"], 
                            min(settings["max_kernel"], 
                                max(w, h) // settings["size_factor"]))
            
            # Ensure odd kernel size
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Apply Gaussian blur to the ROI
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            
            # Replace the original ROI with the blurred version
            blurred_frame[y:y+h, x:x+w] = blurred_roi
        
        return blurred_frame
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection bounding boxes and information on the frame.
        First applies blur to detected areas, then draws overlay information.
        """
        # First, blur the detected areas
        display_frame = self.blur_detections(frame, detections)
        
        # Then draw detection overlays
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
            blur_status = f"[{self.blur_intensity.upper()} BLUR]" if self.enable_blur else "[NO BLUR]"
            info_text = f"Motion {i+1}: Area={int(area)} {blur_status}"
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
        # Add detection count and blur status
        blur_info = f"Blur: {self.blur_intensity.upper()}" if self.enable_blur else "Blur: OFF"
        stats_text = f"Detections: {detection_count} | Frame: {frame_number} | {blur_info}"
        
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
        
        video_ended_normally = False
        
        try:
            while True:
                # Get message from detector
                message = self.input_queue.get()
                
                # Check message type
                if message.get('type') == 'END_OF_VIDEO':
                    print(f"Presenter: Video completed normally after {message['total_frames']} frames")
                    video_ended_normally = True
                    break
                elif message.get('type') == 'INTERRUPTED':
                    print("Presenter: Received interruption signal")
                    break
                elif message.get('type') == 'PROCESSED_FRAME':
                    # Process normal frame
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
                else:
                    # Handle legacy None message or unknown types
                    print("Presenter: Received unknown or legacy end signal")
                    break
                
        except KeyboardInterrupt:
            print("Presenter: Interrupted by user")
        finally:
            if video_ended_normally:
                print("Presenter: Video processing completed successfully")
                # Give user a moment to see the final frame
                print("Presenter: Press any key to close...")
                cv2.waitKey(3000)  # Wait 3 seconds or until key press
            
            self.cleanup()
            print(f"Presenter: Displayed {self.frame_count} total frames")
    
    def cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
        print("Presenter: Cleanup completed")


def run_presenter(input_queue: mp.Queue, enable_blur: bool = True, blur_intensity: str = "medium"):
    """Function to run presenter in a separate process."""
    presenter = Presenter(input_queue, enable_blur, blur_intensity)
    presenter.display_frames()


if __name__ == "__main__":
    # Test the presenter component standalone
    print("Presenter component - run as part of the pipeline system")
