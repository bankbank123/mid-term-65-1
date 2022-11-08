from tkinter import messagebox
import mysql.connector
from tkinter import *
import tkinter as tk
import cv2
from image_analyze import detect_analyze

faceCascada = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
cap = cv2.VideoCapture(0)
def get_data(entry_firstname, entry_lastname):
    root = tk.Tk()
    firstname = str(entry_firstname.get())
    lastname = str(entry_lastname.get())
    list_tuple = (firstname,lastname)
    mydb = mysql.connector.connect(
        host = "localhost",
        user = "root",
        password = "",
        database = "classifier"
    )
    mycursor = mydb.cursor()
    check_id = "SELECT firstname FROM employee WHERE firstname = %s and lastname = %s"
    sql_value = list_tuple
    mycursor.execute(check_id, sql_value)
    myresult = mycursor.fetchall()
    print(myresult)

    if myresult:
        clf.read("classifier/classifier_" + list_tuple[0] + "_" + list_tuple[1] + ".xml")
        img_id = 0   
        root.destroy()
        while (True):
            ret, frame = cap.read()
            frame=detect_analyze(frame, faceCascada, clf, img_id)
            cv2.imshow('frame', frame)
            img_id +=1
            if (cv2.waitKey(1) & 0xFF == ord('s')):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        message = str(myresult) + " already exists"
        messagebox.showerror("error", message)