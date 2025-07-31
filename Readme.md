# Object Detection System - Complete Setup Guide

## Project Overview
This is a comprehensive object detection system that can identify and classify multiple objects in images and video streams using YOLO (You Only Look Once) algorithm. The system supports real-time detection, custom object filtering, and works with both images and videos.

## Features
- ✅ Real-time object detection in images and videos
- ✅ Webcam support for live detection
- ✅ Custom object filtering (detect specific classes only)
- ✅ Adjustable confidence and NMS thresholds
- ✅ Performance metrics (FPS, processing time)
- ✅ Bounding box visualization with confidence scores
- ✅ Support for YOLO v3/v4/v5 models
- ✅ Batch processing capabilities

## Technologies Used
- **Python 3.7+**
- **OpenCV** - Computer vision and image processing
- **NumPy** - Numerical computations
- **YOLO** - Pre-trained object detection model
- **argparse** - Command-line interface

## Installation Instructions

### Step 1: Install Python Dependencies
```bash
pip install opencv-python numpy argparse pathlib
```

### Step 2: Download YOLO Model Files
Create a `yolo` directory in your project folder and download these files:

1. **YOLOv3 Weights** (248 MB)
   ```bash
   wget https://pjreddie.com/media/files/yolov3.weights -P yolo/
   ```

2. **YOLOv3 Configuration**
   ```bash
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -P yolo/
   ```

3. **COCO Class Names**
   ```bash
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -P yolo/
   ```

### Step 3: Project Structure
```
object_detection_project/
│
├── object_detection.py          # Main detection system
├── yolo/
│   ├── yolov3.weights          # YOLO weights file
│   ├── yolov3.cfg              # YOLO configuration
│   └── coco.names              # Class names
├── test_images/                # Sample images for testing
├── test_videos/                # Sample videos for testing
└── results/                    # Output directory
```

## Usage Examples

### 1. Image Detection
```bash
python object_detection.py --mode image --input test_images/city_street.jpg --output results/cst.jpg
```

```bash
python object_detection.py --mode image --input test_images/cat_and_dog.jpg --output results/cdog.jpg
```

```bash
python object_detection.py --mode image --input test_images/breakfast_table.jpg --output results/bfTable.jpg
```

### 2. Real-time Webcam Detection
```bash
python object_detection.py --mode webcam
```

### 3. Video Processing
```bash
python object_detection.py --mode video --input test_videos/t1.mp4 --output results/t1.avi
```

```bash
python object_detection.py --mode video --input test_videos/t2.mp4 --output results/t2.avi
```

```bash
python object_detection.py --mode video --input test_videos/t3.mp4 --output results/t3.avi
```

### 4. Custom Object Detection
```bash
python object_detection.py --mode custom --input test_images/city_street.jpg --target-classes person car bicycle --output results/custom_report.jpg
```

### 5. Adjust Detection Sensitivity
```bash
python object_detection.py --mode image --input city_street.jpg --confidence 0.3 --nms 0.4
```

## Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Detection mode: image, video, webcam, custom | image |
| `--input` | Input image or video path | - |
| `--output` | Output path for results | - |
| `--config` | YOLO config file path | yolo/yolov3.cfg |
| `--weights` | YOLO weights file path | yolo/yolov3.weights |
| `--classes` | Classes names file path | yolo/coco.names |
| `--confidence` | Confidence threshold (0.0-1.0) | 0.5 |
| `--nms` | Non-maximum suppression threshold | 0.4 |
| `--target-classes` | Specific classes to detect | - |

## Supported Object Classes (COCO Dataset)
The system can detect 80 different object classes including:
- **People**: person
- **Vehicles**: car, motorbike, aeroplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Household**: chair, sofa, dining table, toilet, bed, tv, laptop, mouse, remote, keyboard
- **Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- And many more!

## Performance Considerations

### Performance Tips
1. **Lower Resolution**: Resize input images/videos for faster processing
2. **Adjust Confidence**: Higher confidence = fewer false positives, faster processing
3. **Use GPU**: Install OpenCV with CUDA support for GPU acceleration
4. **Model Selection**: YOLOv3 (balance), YOLOv4 (accuracy), YOLOv5 (speed)

## Troubleshooting

### Common Issues and Solutions

1. **"YOLO files not found" Error**
   - Ensure all three YOLO files are downloaded in the `yolo/` directory
   - Check file permissions and paths

2. **Low FPS in Real-time Detection**
   - Reduce input resolution
   - Increase confidence threshold
   - Use GPU acceleration if available

3. **No Objects Detected**
   - Lower confidence threshold (try 0.3 or 0.2)
   - Check if objects are in the supported class list
   - Verify image quality and lighting

4. **ImportError for OpenCV**
   ```bash
   pip install opencv-contrib-python
   ```

## Code Architecture

### Main Components

1. **ObjectDetector Class**
   - Core detection engine
   - Handles YOLO model loading and inference
   - Manages bounding box processing and NMS

2. **Detection Methods**
   - `detect_objects()`: Core detection algorithm
   - `apply_nms()`: Non-maximum suppression
   - `draw_detections()`: Visualization

3. **Processing Modes**
   - `detect_in_image()`: Single image processing
   - `detect_in_video()`: Video/webcam processing
   - `detect_custom_objects()`: Filtered detection


## Learning Outcomes

By working with this project, you'll learn:

1. **Object Detection Algorithms**
   - YOLO architecture and working principles
   - Non-maximum suppression techniques
   - Confidence thresholding

3. **Real-time Image Processing**
   - Video stream handling
   - Frame-by-frame processing
   - Performance monitoring

4. **Computer Vision Techniques**
   - Image preprocessing and blob creation
   - Bounding box calculations
   - Visualization techniques

## Extension Ideas

1. **Object Tracking**: Add object tracking across video frames
2. **Custom UI**: Create a GUI interface using tkinter or PyQt
3. **API Service**: Convert to REST API using Flask/FastAPI
4. **Mobile App**: Port to mobile using frameworks like Kivy
5. **Alert System**: Add notifications for specific object detection
6. **Analytics**: Generate detection reports and statistics

## Resources and References

- [YOLO Official Paper](https://arxiv.org/abs/1506.02640)
- [OpenCV Documentation](https://docs.opencv.org/)
- [COCO Dataset](http://cocodataset.org/)
- [Darknet Framework](https://github.com/pjreddie/darknet)