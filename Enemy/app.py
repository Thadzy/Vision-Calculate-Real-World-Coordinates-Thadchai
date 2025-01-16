import cv2
import numpy as np

def detect_robot_base(image):
    # Convert to HSV colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Stricter thresholds for red color (more saturated and brighter)
    red_lower1 = np.array([0, 150, 100])    # Higher saturation minimum
    red_upper1 = np.array([8, 255, 255])    # Narrower hue range
    red_lower2 = np.array([172, 150, 100])  # Higher saturation minimum
    red_upper2 = np.array([180, 255, 255])  # Narrower hue range

    # Create masks for red
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 + red_mask2

    # Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest circular contour
    best_contour = None
    largest_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Skip tiny contours
        if area < 1000:  # Minimal area threshold
            continue
            
        # Circularity check (perfect circle = 1)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.6:  # Not circular enough
            continue
        
        if area > largest_area:
            largest_area = area
            best_contour = contour

    if best_contour is not None:
        # Get measurements for the best contour
        (x, y), radius = cv2.minEnclosingCircle(best_contour)
        diameter = int(radius * 2)
        
        # Draw the contour and circle
        cv2.drawContours(image, [best_contour], -1, (0, 255, 0), 3)
        cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        
        # Add text annotations
        cv2.putText(image, f"Diameter: {diameter}px", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Process frame
    result = detect_robot_base(frame)
    
    # Display result
    cv2.imshow("Robot Base Detection", result)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()