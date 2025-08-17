#!/usr/bin/env python3
"""
Video Analytics Pipeline System

A multi-process video analytics system with three components:
1. Streamer: Reads video frames from file
2. Detector: Performs motion detection on frames  
3. Presenter: Displays frames with detections, blur effects, and timestamp

Features:
- Real-time motion detection using OpenCV
- Configurable Gaussian blur on detected motion areas
- Multi-intensity blur settings (light, medium, heavy)
- Real-time timestamp and statistics overlay
- Multi-process architecture for optimal performance

Usage:
    python video_analytics_pipeline.py <video_path> [options]
"""

import multiprocessing as mp
import sys
import time
import signal
from pathlib import Path

# Import our components
from streamer import run_streamer
from detector import run_detector
from presenter import run_presenter


class VideoAnalyticsPipeline:
    """
    Main pipeline orchestrator that coordinates all components.
    """
    
    def __init__(self, video_path: str, enable_blur: bool = True, blur_intensity: str = "medium"):
        self.video_path = video_path
        self.enable_blur = enable_blur
        self.blur_intensity = blur_intensity
        self.processes = []
        self.queues = {}
        self.running = False
    
    def setup_queues(self):
        """Create inter-process communication queues."""
        # Queue from Streamer to Detector
        self.queues['streamer_to_detector'] = mp.Queue(maxsize=10)
        
        # Queue from Detector to Presenter
        self.queues['detector_to_presenter'] = mp.Queue(maxsize=10)
        
        print("Pipeline: Queues initialized")
    
    def validate_video_file(self) -> bool:
        """Validate that the video file exists and is accessible."""
        video_file = Path(self.video_path)
        if not video_file.exists():
            print(f"Error: Video file '{self.video_path}' does not exist")
            return False
        
        if not video_file.is_file():
            print(f"Error: '{self.video_path}' is not a file")
            return False
        
        print(f"Pipeline: Video file '{self.video_path}' validated")
        return True
    
    def start_components(self):
        """Start all pipeline components as separate processes."""
        try:
            # Start Streamer process
            streamer_process = mp.Process(
                target=run_streamer,
                args=(self.video_path, self.queues['streamer_to_detector']),
                name="Streamer"
            )
            streamer_process.start()
            self.processes.append(streamer_process)
            print("Pipeline: Streamer process started")
            
            # Start Detector process
            detector_process = mp.Process(
                target=run_detector,
                args=(self.queues['streamer_to_detector'], self.queues['detector_to_presenter']),
                name="Detector"
            )
            detector_process.start()
            self.processes.append(detector_process)
            print("Pipeline: Detector process started")
            
            # Start Presenter process
            presenter_process = mp.Process(
                target=run_presenter,
                args=(self.queues['detector_to_presenter'], self.enable_blur, self.blur_intensity),
                name="Presenter"
            )
            presenter_process.start()
            self.processes.append(presenter_process)
            print(f"Pipeline: Presenter process started (Blur: {'ON' if self.enable_blur else 'OFF'}, Intensity: {self.blur_intensity})")
            
            self.running = True
            print("Pipeline: All components started successfully")
            
        except Exception as e:
            print(f"Pipeline: Error starting components: {e}")
            self.cleanup()
            return False
        
        return True
    
    def monitor_processes(self):
        """Monitor all processes and handle completion/failures."""
        print("Pipeline: Monitoring processes (Press Ctrl+C to stop)")
        
        try:
            while self.running:
                # Check if any process has terminated
                for process in self.processes:
                    if not process.is_alive():
                        print(f"Pipeline: Process {process.name} has terminated")
                        self.running = False
                        break
                
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            print("\nPipeline: Received interrupt signal")
            self.running = False
    
    def cleanup(self):
        """Clean up all processes and resources."""
        print("Pipeline: Starting cleanup...")
        
        # Terminate all processes
        for process in self.processes:
            if process.is_alive():
                print(f"Pipeline: Terminating {process.name}")
                process.terminate()
                process.join(timeout=5)  # Wait up to 5 seconds
                
                # Force kill if still alive
                if process.is_alive():
                    print(f"Pipeline: Force killing {process.name}")
                    process.kill()
                    process.join()
        
        # Clear queues
        for queue_name, queue in self.queues.items():
            try:
                while not queue.empty():
                    queue.get_nowait()
            except:
                pass
        
        print("Pipeline: Cleanup completed")
    
    def run(self):
        """Main pipeline execution method."""
        print("="*60)
        print("Video Analytics Pipeline System")
        print("="*60)
        
        # Validate inputs
        if not self.validate_video_file():
            return False
        
        # Setup infrastructure
        self.setup_queues()
        
        # Start all components
        if not self.start_components():
            return False
        
        # Monitor execution
        self.monitor_processes()
        
        # Cleanup
        self.cleanup()
        
        print("Pipeline: Execution completed")
        return True


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\nReceived signal {signum}")
    sys.exit(0)


def main():
    """Main entry point."""
    import argparse
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Video Analytics Pipeline System with Motion Detection and Blur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_analytics_pipeline.py video.mp4
  python video_analytics_pipeline.py video.mp4 --no-blur
  python video_analytics_pipeline.py video.mp4 --blur-intensity heavy
        """
    )
    
    parser.add_argument(
        "video_path",
        help="Path to the video file to process"
    )
    
    parser.add_argument(
        "--no-blur",
        action="store_true",
        help="Disable blur effect on detected motion areas"
    )
    
    parser.add_argument(
        "--blur-intensity",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="Blur intensity level (default: medium)"
    )
    
    args = parser.parse_args()
    
    # Configure blur settings
    enable_blur = not args.no_blur
    blur_intensity = args.blur_intensity
    
    print(f"Blur Configuration: {'Enabled' if enable_blur else 'Disabled'}")
    if enable_blur:
        print(f"Blur Intensity: {blur_intensity}")
    
    # Create and run pipeline
    pipeline = VideoAnalyticsPipeline(args.video_path, enable_blur, blur_intensity)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()
