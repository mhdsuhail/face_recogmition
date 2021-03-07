import cv2 
import pickle
import pandas as pd
import face_recognition
import numpy as np

face_cascade = cv2.CascadeClassifier('/Users/mhdsuhail/Downloads/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

with open("/Users/mhdsuhail/Desktop/cvv/model.pkl","rb") as f:
    model = pickle.load(f)
#df = pd.read_csv("/Users/mhdsuhail/Desktop/the_data.csv")
#df = df.drop("Unnamed: 0", axis=1)
#print (model.predict(df.iloc[:,0:-1]))
cap = cv2.VideoCapture(0)

while True:
    _,img =cap.read()
    gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_only = img[y:y+h,x:x+w]

        img_encoding = face_recognition.face_encodings(face_only)[0]
        pred_name = model.predict(np.reshape(img_encoding,(1,-1)))
        cv2.putText(img,pred_name[0],(x+5,y+h-5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))

    cv2.imshow('test_window',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#model.predict(np.reshape(img_encoding,(1,-1)))