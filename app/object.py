import cv2
import numpy as np

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Add parameters for line stabilization
previous_lines = []
stability_threshold = 10  # Minimum number of frames a line must be detected
line_memory = 5  # Number of frames to remember
min_line_distance = 30  # Minimum distance between detected lines

def are_lines_similar(line1, line2, threshold=20):
    """Check if two lines are similar in position"""
    x1_1, y1_1, x2_1, y2_1 = line1
    x1_2, y1_2, x2_2, y2_2 = line2
    
    # For vertical lines, compare x coordinates
    if abs(x1_1 - x2_1) < 5 and abs(x1_2 - x2_2) < 5:
        return abs(x1_1 - x1_2) < threshold
    
    # For horizontal lines, compare y coordinates
    if abs(y1_1 - y2_1) < 5 and abs(y1_2 - y2_2) < 5:
        return abs(y1_1 - y1_2) < threshold
    
    return False

def merge_similar_lines(lines):
    """Merge similar lines by averaging their positions"""
    if lines is None or len(lines) == 0:
        return []
    
    merged_lines = []
    used = set()
    
    for i, line1 in enumerate(lines):
        if i in used:
            continue
            
        similar_lines = [line1]
        used.add(i)
        
        for j, line2 in enumerate(lines):
            if j not in used and are_lines_similar(line1[0], line2[0]):
                similar_lines.append(line2)
                used.add(j)
        
        if similar_lines:
            # Average the positions of similar lines
            avg_line = np.mean(similar_lines, axis=0)
            merged_lines.append(avg_line)
    
    return merged_lines

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Convert to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection with adjusted parameters
    edges = cv2.Canny(gray, 30, 150)

    # Dilate edges to connect potential gaps
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Hough Line Transform with adjusted parameters
    lines = cv2.HoughLinesP(edges, 
                           rho=1, 
                           theta=np.pi/180, 
                           threshold=50,
                           minLineLength=100,  # Increased minimum line length
                           maxLineGap=20)      # Increased max gap

    # Get frame dimensions and calculate center
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2

    # Merge similar lines and update previous_lines
    if lines is not None:
        merged_lines = merge_similar_lines(lines)
        previous_lines.append(merged_lines)
        if len(previous_lines) > line_memory:
            previous_lines.pop(0)
    
    detected_positions = []
    
    # Draw stable lines
    if previous_lines:
        stable_lines = []
        for line in previous_lines[-1]:
            x1, y1, x2, y2 = line[0].astype(int)
            
            # Check if the line is vertical or horizontal
            if abs(x1 - x2) < 5:  # Vertical line
                cv2.line(frame, (x1, 0), (x1, frame_height), (0, 255, 0), 2)
                x_position = x1 - center_x
                detected_positions.append(f"X: {x_position}")
                
            elif abs(y1 - y2) < 5:  # Horizontal line
                cv2.line(frame, (0, y1), (frame_width, y1), (0, 255, 0), 2)
                y_position = center_y - y1
                detected_positions.append(f"Y: {y_position}")

    # Draw crosshair
    cv2.line(frame, (center_x, 0), (center_x, frame_height), (0, 0, 255), 2)
    cv2.line(frame, (0, center_y), (frame_width, center_y), (0, 0, 255), 2)
    cv2.putText(frame, "Center: (0,0)", (center_x + 10, center_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display detected positions in top-left corner
    y_offset = 30
    for i, position in enumerate(detected_positions):
        cv2.putText(frame, position, (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Center Line Detection with Coordinates', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()