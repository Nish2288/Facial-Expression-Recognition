import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('models/cnn.hdf5')
EMOTIONS = ["Angry" ,"Disgust","Scared", "Happy", "Sad", "Surprised","Neutral"]

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
       roi_gray = gray[y:y+h, x:x+w]
       roi_gray = cv2.resize(roi_gray, (64, 64))
       roi_gray = roi_gray/255.0
       roi_gray = img_to_array(roi_gray)
       roi_gray = np.expand_dims(roi_gray, axis=0)
       pred = model.predict(roi_gray).argmax(axis = 1)[0]
       print(EMOTIONS[pred])

       
       cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0 , 0), 2)
       cv2.putText(frame, str(EMOTIONS[pred]), (x+40, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2 )
    return frame




video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detect(gray, frame)
    cv2.imshow('frame', face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()