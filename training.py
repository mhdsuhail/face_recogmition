import cv2
import numpy as np 
import os
import json
import pickle
from imutils import paths
from pathlib import Path
import face_recognition
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

c = 0
enc_names = []
cur_dir =os.path.dirname(__file__)
folders=[]
folders = [f for f in os.listdir(cur_dir) if os.path.isdir(os.path.join(cur_dir,f))]
known_patterns = []
names = []
data = {}

for i in folders:
    path = os.path.join(cur_dir,i)
    image_path = list(paths.list_images(path))
    for p in image_path:
        image = face_recognition.load_image_file(p)
        try:
            img_encoding = face_recognition.face_encodings(image)[0]
            known_patterns.append(img_encoding)
            names.append(i)
        except:
            pass
    #data[i] =known_patterns
#print(known_patterns)

"""
with open('data.p','bw') as fp:
    pickle.dump(data,fp)
"""
df = pd.DataFrame(known_patterns)
#df["names"] = names
#print (df)
#df.to_csv("the_data.csv")

re_o = RandomForestClassifier()
re_o.fit(known_patterns,names)

with open("/Users/mhdsuhail/Desktop/cvv/model.pkl","wb") as f:
    pickle.dump(re_o,f)