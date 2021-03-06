import cv2
import os

face_cascade = cv2.CascadeClassifier('/Users/mhdsuhail/Downloads/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
cap  = cv2.VideoCapture(0)
name = input("Enter your name :")
count = 0

folder = os.path.join(os.path.dirname(__file__),name)
if not os.path.exists(name):
    os.makedirs(folder,exist_ok=True)


while True:
    count = count+1
    booo,img =cap.read()
    gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_only = img[y:y+h,x:x+w]

    cv2.imshow('image',img)

    f_path = os.path.join(os.path.dirname(__file__),name,str(count) + '.jpg')
    cv2.imwrite(f_path, face_only)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 150:
        break
cap.release()
cv2.destroyAllWindows()
    



