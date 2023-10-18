import cv2

# Load Haar Cascades for face and smile
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load a video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect smiles within the face ROI
        smiles = smile_cascade.detectMultiScale3(
            roi_gray, scaleFactor=1.8, minNeighbors=20,
            outputRejectLevels=True
        )
        
        # The detectMultiScale3 method returns a tuple with 3 elements:
        # rectangles, reject levels, and level weights (often used as "confidence")
        rectangles, reject_levels, level_weights = smiles
        
        for i, (sx, sy, sw, sh) in enumerate(rectangles):
            confidence = level_weights[i] 
            if confidence > 1.5:  # Adjust threshold as needed
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                cv2.putText(roi_color, f"{confidence:.2f}", (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the output
    cv2.imshow('Smile Detection', frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
