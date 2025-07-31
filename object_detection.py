import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path
import urllib.request
from datetime import datetime

class ObjectDetector:
    def __init__(self, config_path, weights_path, classes_path, confidence_threshold=0.5, nms_threshold=0.4):
        
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Load YOLO network
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Generate random colors for each class
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        print(f"Loaded YOLO model with {len(self.classes)} classes")
        print(f"Output layers: {self.output_layers}")
    
    def detect_objects(self, image):
        """
        Detect objects in an image
        """
        height, width, channels = image.shape
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Set input to the network
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Initialize lists
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids
    
    def apply_nms(self, boxes, confidences, class_ids):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes
        """
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        return indexes
    
    def draw_detections(self, image, boxes, confidences, class_ids, indexes):
        """
        Draw bounding boxes and labels on the image
        """
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                
                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Draw label and confidence
                label_text = f"{label}: {confidence:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw background rectangle for text
                cv2.rectangle(image, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), color, -1)
                
                # Draw text
                cv2.putText(image, label_text, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image
    
    def detect_in_image(self, image_path, output_path=None, show_result=True):
        """
        Detect objects in a single image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        print(f"Processing image: {image_path}")
        
        # Detect objects
        start_time = time.time()
        boxes, confidences, class_ids = self.detect_objects(image)
        
        # Apply NMS
        indexes = self.apply_nms(boxes, confidences, class_ids)
        
        # Draw detections
        result_image = self.draw_detections(image.copy(), boxes, confidences, class_ids, indexes)
        
        processing_time = time.time() - start_time
        
        # Print results
        if len(indexes) > 0:
            print(f"Detected {len(indexes)} objects in {processing_time:.2f} seconds:")
            for i in indexes.flatten():
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                print(f"  - {label}: {confidence:.2f}")
        else:
            print("No objects detected")
        
        # Save result if output path is provided
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        
        # Show result
        if show_result:
            cv2.imshow('Object Detection', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result_image
    
    def detect_in_video(self, video_path, output_path=None, show_result=True):
        """
        Detect objects in video stream (webcam or file) and optionally save the result.
        """
        # Open video source
        if video_path == 0:  # Webcam
            cap = cv2.VideoCapture(0)
            print("Using webcam for real-time detection")
            # If no output_path is provided for webcam, generate a default one
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"results/webcam_output_{timestamp}.avi"
                print(f"Webcam output will be saved to: {output_path}")
        else:
            cap = cv2.VideoCapture(video_path)
            print(f"Processing video: {video_path}")
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Adjust FPS for webcam if it's too high/low for typical recording
        if video_path == 0 and fps == 0: # Some webcams return 0 FPS, set a default
            fps = 20 # A reasonable default for webcam recording
            print(f"Adjusting webcam FPS to default: {fps}")
        elif video_path == 0: # Ensure webcam FPS is within a reasonable range for recording
            fps = min(fps, 30) # Cap webcam FPS to 30 for stable recording
            print(f"Using webcam FPS: {fps}")
        else:
            print(f"Video properties: {width}x{height} @ {fps} FPS")

        # Setup video writer if an output path is provided
        out = None
        if output_path:
            # Ensure the 'results' directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # XVID is a widely supported codec
            # Check if dimensions are valid for video writer
            if width > 0 and height > 0:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    print(f"Warning: Could not create video writer for {output_path}. Output won't be saved.")
                    out = None # Set to None so it's not used
            else:
                print(f"Warning: Invalid video dimensions ({width}x{height}) for writer. Output won't be saved.")


        frame_count = 0
        total_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect objects
            start_time = time.time()
            boxes, confidences, class_ids = self.detect_objects(frame)
            
            # Apply NMS
            indexes = self.apply_nms(boxes, confidences, class_ids)
            
            # Draw detections
            result_frame = self.draw_detections(frame.copy(), boxes, confidences, class_ids, indexes)
            
            processing_time = time.time() - start_time
            total_time += processing_time
            
            # Add FPS info
            fps_display_text = f"FPS: {1/processing_time:.1f}"
            cv2.putText(result_frame, fps_display_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add detection count
            detection_count = len(indexes) if len(indexes) > 0 else 0
            count_text = f"Objects: {detection_count}"
            cv2.putText(result_frame, count_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame if output writer is available
            if out is not None: # Check if 'out' object was successfully created
                out.write(result_frame)
            
            # Show result
            if show_result:
                cv2.imshow('Object Detection - Video', result_frame)
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        if out is not None: # Only release if it was created
            out.release()
        cv2.destroyAllWindows()
        
        # Print summary
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Processed {frame_count} frames")
        print(f"Average FPS: {avg_fps:.2f}")
    
    def detect_custom_objects(self, image_path, target_classes, output_path=None):
        """
        Detect only specific classes of objects
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        print(f"Looking for specific objects: {target_classes}")
        
        # Detect objects
        boxes, confidences, class_ids = self.detect_objects(image)
        
        # Filter for target classes
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        
        for i, class_id in enumerate(class_ids):
            if self.classes[class_id] in target_classes:
                filtered_boxes.append(boxes[i])
                filtered_confidences.append(confidences[i])
                filtered_class_ids.append(class_id)
        
        if filtered_boxes:
            # Apply NMS on filtered results
            indexes = cv2.dnn.NMSBoxes(filtered_boxes, filtered_confidences, 
                                     self.confidence_threshold, self.nms_threshold)
            
            # Draw detections
            result_image = self.draw_detections(image.copy(), filtered_boxes, 
                                              filtered_confidences, filtered_class_ids, indexes)
            
            print(f"Found {len(indexes)} target objects:")
            for i in indexes.flatten():
                label = self.classes[filtered_class_ids[i]]
                confidence = filtered_confidences[i]
                print(f"  - {label}: {confidence:.2f}")
        else:
            print("No target objects found")
            result_image = image
        
        # Save and show result
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        
        cv2.imshow('Custom Object Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return result_image

def download_file(url, filename, description):
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            print(f"\r  Progress: {'█' * (percent // 5):<20} {percent}%", end='', flush=True)
    
    try:
        print(f"\n  Downloading {description}...")
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"\n Failed to download {filename}: {e}")
        return False

def download_yolo_files():
    print("\n Downloading YOLO model files...")
    
    files_to_download = [
        {
            'url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            'filename': 'yolo/yolov3.cfg',
            'description': 'YOLOv3 configuration'
        },
        {
            'url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
            'filename': 'yolo/coco.names',
            'description': 'COCO class names'
        }
    ]
    
    for file_info in files_to_download:
        if not os.path.exists(file_info['filename']):
            if not download_file(file_info['url'], file_info['filename'], file_info['description']):
                return False
        else:
            print(f" File already exists: {file_info['filename']}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Object Detection System')
    parser.add_argument('--mode', choices=['image', 'video', 'webcam', 'custom'], 
                       default='image', help='Detection mode')
    parser.add_argument('--input', type=str, help='Input image or video path')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--config', type=str, default='yolo/yolov3.cfg', 
                       help='YOLO config file path')
    parser.add_argument('--weights', type=str, default='yolo/yolov3.weights', 
                       help='YOLO weights file path')
    parser.add_argument('--classes', type=str, default='yolo/coco.names', 
                       help='Classes names file path')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, 
                       help='NMS threshold')
    parser.add_argument('--target-classes', nargs='+', 
                       help='Specific classes to detect (for custom mode)')
    
    args = parser.parse_args()
    
    # Check if YOLO files exist
    if not all(os.path.exists(f) for f in [args.config, args.weights, args.classes]):
        print("YOLO files not found!")
        download_yolo_files()
        return
    
    try:
        # Initialize detector
        detector = ObjectDetector(args.config, args.weights, args.classes, 
                                args.confidence, args.nms)
        
        if args.mode == 'image':
            if not args.input:
                print("Please provide input image path with --input")
                return
            detector.detect_in_image(args.input, args.output)
            
        elif args.mode == 'video':
            if not args.input:
                print("Please provide input video path with --input")
                return
            detector.detect_in_video(args.input, args.output)
            
        elif args.mode == 'webcam':
            detector.detect_in_video(0)
            
        elif args.mode == 'custom':
            if not args.input or not args.target_classes:
                print("Please provide input image and target classes")
                return
            detector.detect_custom_objects(args.input, args.target_classes, args.output)
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure YOLO files are properly downloaded and paths are correct")


if __name__ == "__main__":
    # Example usage without command line arguments
    print("Object Detection System")
    print("=====================")
    
    # Check if running with command line arguments
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Demo mode - show instructions
        download_yolo_files()