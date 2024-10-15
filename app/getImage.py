import cv2
from time import sleep

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(2)
sleep(2)

image_counter = 0

while True:
    try:
        check, frame = webcam.read()
        print(check)  # prints true as long as the webcam is running
        print(frame)  # prints matrix values of each frame
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            # Save the image
            filename = f'saved_img_{image_counter}.jpg'
            cv2.imwrite(filename=filename, img=frame)
            print(f"Image saved as {filename}")
        
            
            # Increment the counter for the next image
            image_counter += 1
            
            # Continue the loop without breaking
            continue
        
        elif key == ord('q'):
            break
    
    except KeyboardInterrupt:
        break

print("Turning off camera.")
webcam.release()
print("Camera off.")
print("Program ended.")
cv2.destroyAllWindows()