import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque
import threading
import queue
import time

class BallDetector:
    def __init__(self, model_path, camera_matrix, dist_coeffs, conf_threshold=0.5, iou_threshold=0.45):
        # Initialize the ball detector
        self.model = YOLO(model_path)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.known_diameter = 180  # mm
        self.camera_height = 300   # mm from ground
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_thresholdhttps://github.com/Thadzy/Vision-Calculate-Real-World-Coordinates-Thadchai.git
        
        # Define colors for different classes
        self.colors = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'purple': (128, 0, 128),
            'default': (0, 255, 0)
        }
        
        # Default target color for detection
        self.target_color = 'default'
        
        # Initialize threading and buffers
        self.frame_buffer = queue.Queue(maxsize=2)
        self.result_buffer = queue.Queue(maxsize=2)
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Cache for visualization
        self.grid_overlay = None
        self.grid_cache = {}
        self.text_cache = {}
        
        # Performance monitoring
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
    def set_ball_color(self, color):
        """Set the target color for ball detection."""
        if color in self.colors:
            self.target_color = color
            print(f"Ball color set to {color}.")
        else:
            print(f"Color '{color}' not recognized. Available colors: {', '.join(self.colors.keys())}")
    
    # The rest of your methods are unchanged...
    def __del__(self):
        """Cleanup resources"""
        self.processing_active = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)

    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        while self.processing_active:
            try:
                # Get frame with timeout to allow checking processing_active
                frame = self.frame_buffer.get(timeout=0.1)
                
                # Flip frame once
                flipped_frame = cv2.flip(frame, -1)
                
                # Batch process with YOLO
                results = self.model(flipped_frame, conf=self.conf_threshold, iou=self.iou_threshold)
                
                # Process all detections in batch
                processed_results = self._batch_process_detections(results[0], flipped_frame)
                
                # Store results
                self.result_buffer.put((processed_results, flipped_frame))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue

    def _batch_process_detections(self, results, frame):
        """Process all detections in batch for better performance."""
        processed_results = []
        
        if len(results.boxes) > 0:
            boxes = results.boxes
            
            # Extract all coordinates at once
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            
            # Batch calculate centers and diameters
            centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
            centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
            diameters_pixels = np.maximum(xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1])
            
            # Batch calculate Z coordinates using the known diameter and focal length
            focal_length = self.camera_matrix[0, 0]
            Z = ((self.known_diameter * focal_length) / diameters_pixels)
            
            # Batch calculate normalized coordinates
            normalized_x = (centers_x - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
            normalized_y = (centers_y - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
            
            # Batch calculate real world coordinates (X, Y)
            X = normalized_x * Z
            Y = -normalized_y * Z + self.camera_height
            
            # Estimate real-world diameter using calculated Z
            estimated_diameter = (diameters_pixels * Z) / focal_length
            
            # Create results
            for i in range(len(boxes)):
                processed_results.append({
                    'x': float(X[i]),
                    'y': float(Y[i]),
                    'z': float(Z[i]),
                    'diameter': float(estimated_diameter[i]),
                    'confidence': float(confidences[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': self.model.names[int(class_ids[i])].lower(),
                    'pixel_bbox': tuple(map(int, xyxy[i])),
                    'pixel_center': (int(centers_x[i]), int(centers_y[i]))
                })
        
        return processed_results

    def process_frame(self, frame):
        """Non-blocking frame processing with FPS calculation"""
        current_time = time.time()
        self.fps_history.append(1 / (current_time - self.last_frame_time))
        self.last_frame_time = current_time
        
        # Add new frame to buffer if not full
        if not self.frame_buffer.full():
            self.frame_buffer.put(frame)
        
        try:
            return self.result_buffer.get_nowait()
        except queue.Empty:
            return None, None

    def _create_grid_overlay(self, shape):
        """Create and cache grid overlay"""
        cache_key = f"{shape[0]}_{shape[1]}"
        
        if cache_key not in self.grid_cache:
            height, width = shape[:2]
            overlay = np.zeros(shape, dtype=np.uint8)
            
            # Create grid lines using vectorization
            x_coords = np.arange(0, width, 50)
            y_coords = np.arange(0, height, 50)
            
            for x in x_coords:
                cv2.line(overlay, (x, 0), (x, height), (50, 50, 50), 1)
            for y in y_coords:
                cv2.line(overlay, (0, y), (width, y), (50, 50, 50), 1)
                
            self.grid_cache[cache_key] = overlay
        
        return self.grid_cache[cache_key].copy()

    def _get_cached_text(self, text, scale):
        """Cache rendered text for reuse"""
        cache_key = f"{text}_{scale}"
        if cache_key not in self.text_cache:
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            self.text_cache[cache_key] = (w, h)
        return self.text_cache[cache_key]

    def draw_results(self, frame, results):
        """Draw bounding boxes and labels for detected objects."""
        if frame is None or results is None:
            return frame

        height, width = frame.shape[:2]

        # Apply cached grid overlay
        overlay = self._create_grid_overlay(frame.shape)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Draw FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)

        # Draw detections
        for result in results:
            color = self.colors.get(result['class_name'], self.colors['default'])
            if result['class_name'] != self.target_color:
                continue  # Skip drawing if the detected color doesn't match target

            x1, y1, x2, y2 = result['pixel_bbox']
            center_x, center_y = result['pixel_center']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (center_x, center_y), 5, color, -1)

            # Draw labels
            label = f"{result['class_name'].upper()} {result['confidence']:.2f}"
            coords_text = f"X:{result['x']:.0f} Y:{result['y']:.0f} Z:{result['z']:.0f}"
            diameter_text = f"Diameter: {result['diameter']} mm"

            label_w, label_h = self._get_cached_text(label, 0.6)
            coords_w, coords_h = self._get_cached_text(coords_text, 0.5)
            diameter_w, diameter_h = self._get_cached_text(diameter_text, 0.5)

            cv2.rectangle(frame, (x1, y1-label_h-5), (x1+label_w, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, coords_text, (x1, y2+coords_h+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, diameter_text, (x1, y2+coords_h+diameter_h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

def main():
    """Main function with color selection feature."""
    # Camera and model parameters
    camera_matrix = np.array([[470.52148559, 0., 297.74845784],
                             [0., 471.19349701, 230.78254753],
                             [0., 0., 1.]])
    dist_coeffs = np.array([0.22630361, -0.49591966, -0.00215079, -0.00356293, 0.55533988])

    # Initialize detector
    detector = BallDetector(
        model_path=r"D:\Thadzy\ABU\Vision-Calculate-Real-World-Coordinates-Thadchai\new.pt",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        conf_threshold=0.6,
        iou_threshold=0.45
    )

    # User input for color selection
    color = input("Enter the ball color you want to detect (red, blue, purple, default): ").strip().lower()
    detector.set_ball_color(color)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Controls:")
    print("Press 'q' to quit")
    print("Press '+'/'-' to adjust confidence threshold")
    print("Press '['/']' to adjust IoU threshold")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
                
            # Process and display results
            results, processed_frame = detector.process_frame(frame)
            if processed_frame is not None:
                display_frame = detector.draw_results(processed_frame, results)
                cv2.imshow('Ball Detection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                detector.conf_threshold = min(1.0, detector.conf_threshold + 0.05)
            elif key == ord('-'):
                detector.conf_threshold = max(0.0, detector.conf_threshold - 0.05)
            elif key == ord(']'):
                detector.iou_threshold = min(1.0, detector.iou_threshold + 0.05)
            elif key == ord('['):
                detector.iou_threshold = max(0.0, detector.iou_threshold - 0.05)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
