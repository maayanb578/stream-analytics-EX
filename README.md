# Video Analytics Pipeline - Usage Examples

This document provides comprehensive examples for using the Video Analytics Pipeline System with motion detection and blur capabilities.

## System Overview

The pipeline consists of three components running in separate processes:
- **Streamer**: Reads video frames from file
- **Detector**: Performs motion detection using OpenCV
- **Presenter**: Displays video with blur effects and overlays

## Basic Usage

### Default Configuration (Medium Blur)
```bash
python3 video_analytics_pipeline.py "People - 6387.mp4"
```
- Enables medium intensity blur on detected motion areas
- Shows green bounding boxes around motion
- Displays real-time timestamp and statistics

### Display Help
```bash
python3 video_analytics_pipeline.py --help
```

## Blur Configuration Examples

### Disable Blur (Maximum Performance)
```bash
python3 video_analytics_pipeline.py "People - 6387.mp4" --no-blur
```
- No blur processing for optimal performance
- Still shows motion detection with bounding boxes
- Status bar shows "Blur: OFF"

### Light Blur (Subtle Effect)
```bash
python3 video_analytics_pipeline.py "People - 6387.mp4" --blur-intensity light
```
- Minimal blur with small kernel sizes (9-21px)
- Good for subtle privacy effects
- Lower computational overhead

### Medium Blur (Default - Balanced)
```bash
python3 video_analytics_pipeline.py "People - 6387.mp4" --blur-intensity medium
```
- Moderate blur with medium kernel sizes (15-35px)
- Good balance between effect and performance
- Default setting if no intensity specified

### Heavy Blur (Strong Effect)
```bash
python3 video_analytics_pipeline.py "People - 6387.mp4" --blur-intensity heavy
```
- Strong blur with large kernel sizes (25-51px)
- Maximum privacy effect
- Higher computational cost

## Running Individual Components

### Test Streamer Only
```bash
python3 streamer.py "People - 6387.mp4"
```

### Test Basic Motion Detection (Original)
```bash
python3 basic_vmd.py
```

## Controls During Playback

- **ESC key**: Quit the application
- **'q' key**: Quit the application  
- **Ctrl+C**: Force stop all processes

## Output Information

### Visual Elements
- **Green rectangles**: Motion detection bounding boxes
- **Red dots**: Center points of detected motion
- **White timestamp**: Current date/time (upper left)
- **Yellow statistics**: Detection count, frame number, blur status (bottom left)
- **Detection labels**: Show motion number, area, and blur status

### Console Output
```
Pipeline: Video file 'People - 6387.mp4' validated
Pipeline: Queues initialized
Pipeline: Streamer process started
Pipeline: Detector process started
Pipeline: Presenter process started (Blur: ON, Intensity: medium)
Pipeline: All components started successfully
```

## File Structure

After running the system, your directory should contain:
```
EX/
├── video_analytics_pipeline.py    # Main pipeline orchestrator
├── streamer.py                    # Video frame streaming component
├── detector.py                    # Motion detection component
├── presenter.py                   # Display and blur component
├── basic_vmd.py                   # Original simple motion detection
├── People - 6387.mp4             # Your video file
└── usage_example.md               # This file
```

## Performance Tips

1. **For maximum performance**: Use `--no-blur`
2. **For real-time processing**: Use `--blur-intensity light`
3. **For privacy applications**: Use `--blur-intensity heavy`
4. **Large video files**: The system automatically handles frame rate control
5. **Memory usage**: Queue sizes are limited to prevent memory issues

## Troubleshooting

### Video File Issues
- Ensure video file exists and is readable
- Supported formats: MP4, AVI, MOV, and other OpenCV-supported formats
- Check file path is correct (use absolute paths if needed)

### Performance Issues
- Reduce blur intensity or disable blur
- Close other applications to free up CPU/memory
- Ensure video file is stored locally (not network drive)

### Display Issues
- Ensure X11 forwarding if using SSH
- Check OpenCV GUI backend is properly installed
- Try different video files to isolate issues

## Example Command Variations

```bash
# Full path to video file
python3 video_analytics_pipeline.py "/home/user/videos/sample.mp4" --blur-intensity heavy

# Different video formats
python3 video_analytics_pipeline.py "sample.avi" --no-blur
python3 video_analytics_pipeline.py "sample.mov" --blur-intensity light

# Using relative paths
python3 video_analytics_pipeline.py "./videos/test.mp4"
```

## System Requirements

- Python 3.7+
- OpenCV (cv2)
- imutils
- numpy
- multiprocessing support
- Display system (X11/Wayland for GUI)
