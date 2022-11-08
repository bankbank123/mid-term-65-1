import cv2

#img = cv2.imread("image.jpg", 0)
#cv2.imshow('Show Result', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite('result.png', img)

faceCascada = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def create_dataset(img, id, img_id, folder):
    cv2.imwrite("data/" + str(folder) +"/pic"+ str(id)+ "." + str(img_id) + ".jpg", img)

def draw_boundary_data(img, classifier, scaleFactor, minNeighbors, color, text):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coords = []

    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 5)
        cv2.putText(img, text, (x,y-4),cv2.FONT_HERSHEY_SIMPLEX,1,color,1)
        coords=[x,y,w,h]
    return img, coords

def detect_data(img, faceCascada, img_id, id):
    img,coords = draw_boundary_data(img, faceCascada, 1.1, 30, (0, 0, 255), "Face")
    if len(coords) == 4:
            #img(y:y+H),(x:x+w) 
            folder = id
            result = img[coords[1]: coords[1] + coords[3], coords[0]:coords[0]+coords[2]]
            create_dataset(result, id, img_id, folder)
    
    return img





