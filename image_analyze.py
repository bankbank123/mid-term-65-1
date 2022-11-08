import cv2

def create_dataset(img, id, img_id):
    cv2.imwrite("data_analyze/id2/pic." + str(id)+ "." + str(img_id) + ".jpg", img)

def draw_boundary_analyze(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coords = []

    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 5)
        id,con = clf.predict(gray[y:y+h, x:x+w])
        
        if con <= 40 :
            cv2.putText(img, "Chidchanun", (x,y-4),cv2.FONT_HERSHEY_SIMPLEX,1,color,1)
        else:
            cv2.putText(img, "Unknow", (x,y-4),cv2.FONT_HERSHEY_SIMPLEX,1,color,1)
            

        if (con < 40):
            con = "     {0}%".format(round(100 - con))
        else:
            con = "     {0}%".format(round(100 - con))

        print(str(con))
        coords=[x,y,w,h]

    return img, coords

def detect_analyze(img, faceCascada, clf, img_id):
    img, coords = draw_boundary_analyze(img, faceCascada, 1.1, 20, (0, 0, 255), clf)
    if len(coords) == 4:
            #img(y:y+H),(x:x+w) 
            id = 2
            result = img[coords[1]: coords[1] + coords[3], coords[0]:coords[0]+coords[2]]
            create_dataset(result, id, img_id)
    
    
    return img

