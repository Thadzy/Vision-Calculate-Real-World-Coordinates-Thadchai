import cv2
import numpy as np

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2  # Center point (0,0)

    # Draw the cross section (+) in the middle of the frame (as reference)
    cv2.line(frame, (center_x, 0), (center_x, frame_height), (0, 0, 0), 2)  # Vertical line
    cv2.line(frame, (0, center_y), (frame_width, center_y), (0, 0, 0), 2)   # Horizontal line

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range to track (for green color in your case)
    lower_green = np.array([35, 100, 100])  # Adjust based on your green target
    upper_green = np.array([85, 255, 255])
    
    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours of the object
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # If a contour is detected, draw it and find the center
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        if radius > 10:  # Filter based on radius
            # Draw the circle on the object
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

            # Calculate the position relative to the center of the cross (0, 0)
            relative_x = int(x) - center_x
            relative_y = int(y) - center_y

            # Display the relative coordinates
            cv2.putText(frame, f"Position: ({relative_x}, {relative_y})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Object Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
