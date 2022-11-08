import cv2

from training import train_classifier
from image_create_data import detect_data
from image_analyze import detect_analyze

faceCascada = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier_bank.xml")
cap = cv2.VideoCapture("1.mp4")

def main():
    print("1.Create_data")
    print("2.Analyze")
    print("3.traning")
    print("0.exit")
    choose = int(input("Enter you number : "))

    if choose == 1:
    
        img_id = 0
        while (True):
            ret, frame = cap.read()
            frame=detect_data(frame, faceCascada, img_id)
            cv2.imshow('frame', frame)
            img_id +=1
            if (cv2.waitKey(1) & 0xFF == ord('s')):
                break
        cap.release()
        cv2.destroyAllWindows()
        main()
            
    elif choose == 2:
        
        while (True):
            ret, frame = cap.read()
            frame=detect_analyze(frame, faceCascada, clf)
            cv2.imshow('frame', frame)
        
            if (cv2.waitKey(1) & 0xFF == ord('s')):
                break
        cap.release()
        cv2.destroyAllWindows()
        main()
        

    elif choose == 3:

        train_classifier("data")
        main()
    else:
        print("End the programs")

main()
