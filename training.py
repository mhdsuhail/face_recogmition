import cv2
import numpy as np 
import os
import json
from imutils import paths
from pathlib import Path
import face_recognition

c = 0
enc_names = []
cur_dir =os.path.dirname(__file__)
folders=[]
folders = [f for f in os.listdir(cur_dir) if os.path.isdir(os.path.join(cur_dir,f))]
known_patterns = []
data = {}

for i in folders:
    path = os.path.join(cur_dir,i)
    image_path = list(paths.list_images(path))
    for p in image_path:
        image = face_recognition.load_image_file(p)
        try:
            img_encoding = face_recognition.face_encodings(image)[0]
            known_patterns.append(img_encoding)
        except:
            pass
    data[i] =known_patterns
    
with open('data.json','w') as fp:
    json.dump(data,fp)



    
    


        