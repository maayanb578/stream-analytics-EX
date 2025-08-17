import cv2
import imutils
import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class Detector:
    """
    Detector component that performs motion detection on video frames.
    """
    
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.prev_frame = None
        self.frame_count = 0
    
    def detect_motion(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect motion in the current frame compared to the previous frame.
        Returns a list of detection dictionaries with bounding boxes and areas.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If this is the first frame, store it and return no detections
        if self.prev_frame is None:
            self.prev_frame = gray_frame
            return []
        
        # Calculate difference between current and previous frame
        diff = cv2.absdiff(gray_frame, self.prev_frame)
        
        # Apply threshold to get binary image
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Process contours to create detection objects
        detections = []
        for contour in contours:
            # Filter out small contours (noise)
            area = cv2.contourArea(contour)
            if area < 500:  # Minimum area threshold
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create detection object
            detection = {
                'bbox': (x, y, w, h),  # Bounding box (x, y, width, height)
                'area': area,
                'center': (x + w // 2, y + h // 2),
                'contour': contour.tolist()  # Convert to list for serialization
            }
            detections.append(detection)
        
        # Update previous frame
        self.prev_frame = gray_frame
        
        return detections
    
    def process_frames(self):
        """Main processing loop that receives frames and detects motion."""
        print("Detector: Starting motion detection processing")
        
        try:
            while True:
                # Get message from streamer
                message = self.input_queue.get()
                
                # Check for end signal
                if message is None:
                    print("Detector: Received end signal")
                    break
                
                frame = message['frame']
                frame_number = message['frame_number']
                timestamp = message['timestamp']
                
                # Perform motion detection
                detections = self.detect_motion(frame)
                
                # Create output message
                output_message = {
                    'frame': frame,
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': detections,
                    'detection_count': len(detections)
                }
                
                # Send to presenter
                self.output_queue.put(output_message)
                
                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    print(f"Detector: Processed {self.frame_count} frames")
                
        except KeyboardInterrupt:
            print("Detector: Interrupted by user")
        finally:
            # Send end signal to presenter
            self.output_queue.put(None)
            print("Detector: Processing completed")


def run_detector(input_queue: mp.Queue, output_queue: mp.Queue):
    """Function to run detector in a separate process."""
    detector = Detector(input_queue, output_queue)
    detector.process_frames()


if __name__ == "__main__":
    # Test the detector component standalone
    print("Detector component - run as part of the pipeline system")
