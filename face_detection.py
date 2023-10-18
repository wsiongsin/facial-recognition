import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
detector = MTCNN()
alpha = 0.7 
prev_boxes = np.array([])  # Store previous bounding boxes


while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces
    faces = detector.detect_faces(frame)
    
    # Draw rectangle around the faces
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the output
    cv2.imshow('frame', frame)
    
    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(faces) > 0:
        current_boxes = faces[0]  # Adjust according to actual structure of `faces`
        
        print(type(current_boxes), current_boxes)  # Debug print
        
        # Check if the keys exist and the type of `current_boxes` is dict
        if (isinstance(current_boxes, dict) and 
            all(key in current_boxes for key in ['x', 'y', 'w', 'h'])):

            if not prev_boxes:
                prev_boxes = current_boxes
            else:
                # Calculate running average for each coordinate separately
                current_boxes['x'] = int(alpha * current_boxes['x'] + (1 - alpha) * prev_boxes['x'])
                current_boxes['y'] = int(alpha * current_boxes['y'] + (1 - alpha) * prev_boxes['y'])
                current_boxes['w'] = int(alpha * current_boxes['w'] + (1 - alpha) * prev_boxes['w'])
                current_boxes['h'] = int(alpha * current_boxes['h'] + (1 - alpha) * prev_boxes['h'])
                prev_boxes = current_boxes
        else:
            print("Unexpected structure for `current_boxes`.")
# Release resources
cap.release()
cv2.destroyAllWindows()
