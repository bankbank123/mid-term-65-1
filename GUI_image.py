import os
from tkinter import *
import webbrowser
from GradientFrame import GradientFrame
import cv2
import tkinter as tk
from get_data import get_data
from training import train_classifier
from image_create_data import detect_data
from image_analyze import detect_analyze
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector

def app():
    root = tk.Tk()
    root.title("Image processing")
    canvas = GradientFrame(root,height=3000, width=3000)
    canvas.pack()
    faceCascada = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    cap = cv2.VideoCapture(0)

    def check_database():
        webbrowser.open('http://localhost/project_web/database.php', new=2)

    def insert_database(firstname, lastname):    
        mydb = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = "",
            database = "classifier"
        )
        
        mycursor = mydb.cursor()
        name_check = "SELECT firstname,lastname FROM employee WHERE firstname = %s and lastname = %s"
        name_check_val = (firstname, lastname)
        mycursor.execute(name_check, name_check_val)
        myresult = mycursor.fetchall()
        
        if myresult:
            message = str(myresult) + " already exists"
            messagebox.showerror("error", message)

        else:
            sql_insert = "INSERT INTO employee (firstname, lastname, filename) VALUES (%s, %s, %s)"
            file_name = firstname + "class" + str(mycursor.lastrowid)
            sql_info = (firstname, lastname, file_name)
            print(file_name)
            mycursor.execute(sql_insert, sql_info)
            mydb.commit()
            messagebox.showinfo("info", "1 record inserted, ID:" + str(mycursor.lastrowid))
            #sql_insert_file = "INSERT INTO employee (filename) VALUES (%s)"
            #mycursor.execute(sql_insert_file, file_name)
            train_classifier("data/" + str(mycursor.lastrowid), firstname + "1" )
            

    def update_database():
        mydb = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = "",
            database = "classifier"
        )
        mycursor = mydb.cursor()

    def create_data():
        data = tk.Toplevel(root)
        data.title("create data")
        canvas = tk.Canvas(data, height=500, width=800)
        canvas.pack()
        Label_id = tk.Label(data, text="ID" ,font=20)
        Label_id.place(relx=0.3, rely=0)
        entry_id = tk.Entry(data, font=20)
        entry_id.place(relx=.35, rely=0, relheight=0.05, relwidth=0.3)
        button_id = tk.Button(data, text="comfirm", command=lambda:ids(), font=20)
        button_id.place(relx=0.6, rely=0, relheight=0.05, relwidth=0.1)

        def ids():
            id = str(entry_id.get())
            directory = id
            parent_dir = "C:/Users/chidc/OneDrive/Desktop/image process/data"
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)
            img_id = 0
            cap = cv2.VideoCapture(0)
            while (True):
                ret, frame = cap.read()
                frame=detect_data(frame, faceCascada, img_id, id)
                cv2.imshow('frame', frame)
                img_id +=1
                if (cv2.waitKey(1) & 0xFF == ord('s')):
                    break
            data.destroy()
            cap.release()
            cv2.destroyAllWindows()
        
                
    def analyze():
        data = tk.Toplevel(root)
        data.title("analyze")
        canvas = tk.Canvas(data, height=500, width=800)
        canvas.pack()
        Label_firstname = tk.Label(data, text="Firstname" ,font=20)
        Label_firstname.place(relx=0.085, rely=0.025)
        entry_firstname = tk.Entry(data,font=20)
        entry_firstname.place(relx=0.2 , rely= 0.025, relwidth=.6, relheight=.05)
        Label_lastname = tk.Label(data, text="Lastname" ,font=20)
        Label_lastname.place(relx=0.085, rely=0.1)
        entry_lastname = tk.Entry(data,font=20)
        entry_lastname.place(relx=0.2 , rely= 0.1, relwidth=.6, relheight=.05)
        button_firstname = tk.Button(data, text="Confirm", command=lambda:get_data(entry_firstname, entry_lastname) ,font=18)
        button_firstname.place(relx=0.45, rely=0.175, relwidth=0.1,relheight=0.05)

    def training():
        training = tk.Toplevel(root)
        training.title("traning Image")
        canvas = tk.Canvas(training, height=500, width=800)
        canvas.pack()
        Label_firstname = tk.Label(training, text="Firstname" ,font=20)
        Label_firstname.place(relx=0.085, rely=0.025)
        entry_firstname = tk.Entry(training,font=20)
        entry_firstname.place(relx=0.2 , rely= 0.025, relwidth=.6, relheight=.05)
        Label_lastname = tk.Label(training, text="Lastname" ,font=20)
        Label_lastname.place(relx=0.085, rely=0.1)
        entry_lastname = tk.Entry(training,font=20)
        entry_lastname.place(relx=0.2 , rely= 0.1, relwidth=.6, relheight=.05)
        button_firstname = tk.Button(training, text="Confirm", command=lambda:get_data() ,font=18)
        button_firstname.place(relx=0.45, rely=0.175, relwidth=0.1,relheight=0.05)
        

        def get_data():
            firstname = entry_firstname.get()
            lastname = entry_lastname.get()
            #list_tuple = (firstname, lastname)
            insert_database(firstname, lastname)

    def show_img():
        img = ImageTk.PhotoImage(Image.open("data_analyze/id2/pic.2.1.jpg"))
        panel = tk.Label(root,  image=img)
        panel.place(relx=0.5 , rely=0.5)

    face_photo = PhotoImage(file="ai.png")
    face_label = tk.Label(image=face_photo) 
    face_label.place(relx=.5, rely=0, relheight=1, relwidth=0.5)
    button_data = tk.Button(text="create data", command=create_data, font=30)
    button_data.place(relx=0 , rely=0, relheight=0.125, relwidth=0.2)
    button_anlyze = tk.Button(text="analyze", command=analyze, font=30)
    button_anlyze.place(relx=0 , rely=0.15, relheight=0.125, relwidth=0.2)
    button_training = tk.Button(text="traning", command=training, font=30)
    button_training.place(relx=0 , rely=0.3, relheight=0.125, relwidth=0.2)
    button_show = tk.Button(text="show image", command=show_img, font=30)
    button_show.place(relx=.225 , rely=0, relheight=0.125, relwidth=0.2)
    button_url = tk.Button(text="Database", command=check_database, font=30)
    button_url.place(relx=.225 , rely=.15, relheight=0.125, relwidth=0.2)
    button_exit = tk.Button(text="exit", command=root.destroy, font=30)
    button_exit.place(relx=.225 , rely=0.3, relheight=0.125, relwidth=0.2)
    
    root.mainloop()

app()
        
