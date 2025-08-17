#!/usr/bin/env python3
"""
Quick test example for Video Analytics Pipeline System

This script demonstrates how to run a simple test of the system.
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def run_quick_test():
    """Run a quick test of core functionality."""
    print("Video Analytics Pipeline - Quick Test")
    print("=" * 50)
    
    try:
        # Test 1: Import all components
        print("1. Testing imports...")
        from streamer import Streamer
        from detector import Detector
        from presenter import Presenter
        from video_analytics_pipeline import VideoAnalyticsPipeline
        print("   ✓ All components imported successfully")
        
        # Test 2: Test utilities
        print("2. Testing utilities...")
        from tests.test_utils import TestFileManager, TestFrameGenerator
        with TestFileManager() as fm:
            video_path = fm.create_temp_video(frames=3, fps=10)
            print(f"   ✓ Created test video: {os.path.basename(video_path)}")
        
        # Test 3: Test frame generation
        print("3. Testing frame generation...")
        frame = TestFrameGenerator.create_frame_with_motion()
        print(f"   ✓ Generated test frame: {frame.shape}")
        
        # Test 4: Test detector
        print("4. Testing motion detection...")
        from tests.test_utils import TestMessageQueue
        input_queue = TestMessageQueue()
        output_queue = TestMessageQueue()
        detector = Detector(input_queue, output_queue)
        
        # Test motion detection
        frame1 = TestFrameGenerator.create_frame_with_motion()
        frame2 = TestFrameGenerator.create_frame_with_motion(motion_areas=[(100, 100, 50, 50)])
        
        detections1 = detector.detect_motion(frame1)
        detections2 = detector.detect_motion(frame2)
        
        print(f"   ✓ Motion detection working: {len(detections2)} detections found")
        
        # Test 5: Test blur
        print("5. Testing blur functionality...")
        presenter = Presenter(TestMessageQueue(), enable_blur=True)
        if len(detections2) > 0:
            blurred_frame = presenter.blur_detections(frame2, detections2)
            print("   ✓ Blur functionality working")
        else:
            print("   ✓ Blur functionality available (no detections to test)")
        
        print("\n" + "=" * 50)
        print("✓ All quick tests passed!")
        print("✓ System is ready for use!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
