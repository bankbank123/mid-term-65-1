import numpy as np
from PIL import Image
import os, cv2
import shutil

def train_classifier(data_dir, name):
    path = [os.path.join (data_dir,f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img=Image.open(image).convert("L")
        imageNp=np.array(img, 'uint8')
        id=int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier_" + str(name) + ".xml")
    src_path = "C:/Users/chidc/OneDrive/Desktop/image process" + "/classifier_" + str(name) + ".xml"
    dst_path = "C:/Users/chidc/OneDrive/Desktop/image process/classifier"
    shutil.move(src_path, dst_path)