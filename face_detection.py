import cv2

# Pre-trained model data
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#opening default webcam (set parameter to '1' if externally connected a webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect face after processing grayscale frames
    faces = face_cascade.detectMultiscale(gray, scaleFactor=1.3, minNeighbors=5)

    #draw box around face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    #show final output opening dedicated window
    cv2.imshow('Face Detection', frame)

    #break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#quit webcam and release windows
cap.release()
cv2.destroAllwindows()
