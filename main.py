import cv2
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from datetime import date, datetime
import os
import pandas as pd
import pyodbc
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pymongo
from bson.binary import Binary

# Define class
classname = np.array(["Obama_2", "Trump_1", "Putin_3", "unknown_0"])

# Load trained model
my_model = load_model('model.h5')
facedetec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# http server
app = Flask(__name__)

#Saving Date today
def datetoday():
    return date.today().strftime("%m_%d_%y")

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if f'Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday()}.csv', 'w') as f:
        f.write('Name,Id,Time')

# Add attendance of a specific user
def add_attendence(name, id):
    username = name
    userid = id
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    if name not in list(df['Name']):
        with open(f'Attendance/Attendance-{datetoday()}.csv', 'a') as f2:
            f2.write(f'\n{username},{userid},{current_time}')
            return True
    return False

# Add attendance of a specific user to database SQL
def MakeAttendance(Id, Name):
    conn = pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=DESKTOP-58TVC1M\\SQLEXPRESS;"
        "Database=FaceRecognition;"
        "Trusted_connection=yes"
    )

    cursor = conn.cursor() # create connection between database and python

    sql_query = 'select * from Attendance'
    df = pd.read_sql(sql = sql_query, con = conn)
    print(Id, Name)

    TimeIn = datetime.now().time()  # current time
    DateIn = datetime.now().date()  # current date

    if Id not in list(df['UserId']):
        #cursor.execute("SET IDENTITY_INSERT Attendance ON")
        cursor.execute('insert into FaceRecognition.dbo.Attendance (FullName, DateIn, TimeIn, UserId) values (?, ?, ?, ?)', (Name, DateIn, TimeIn, Id))
        #cursor.execute("SET IDENTITY_INSERT Attendance OFF")

        conn.commit()

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/start', methods=['GET'])
def start():
    video = cv2.VideoCapture(0)
    facedetec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while 1:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetec.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(src=roi_gray, dsize=(100, 100))
            roi_gray = roi_gray.reshape((100, 100, 1))
            roi_gray = np.array(roi_gray)
            result = my_model.predict(np.array([roi_gray]))[0]

            if result[np.argmax(result)] >= 0.5:
                label = classname[np.argmax(result)]
                parse = label.split("_")
                name = parse[0]
                id = parse[1]
                MakeAttendance(id, name)
                #add_attendence(name, id)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 2)
        cv2.imshow('Frame', frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    return None


@app.route('/add', methods=['GET', 'POST'])
def add():
    path_dataset = "dataset\\"

    newusername = request.form['newusername']
    newuserid = request.form['newuserid']

    userfolder = path_dataset + newusername + "_" + str(newuserid)
    if not os.path.isdir(userfolder):
        os.makedirs(userfolder)

    cap = cv2.VideoCapture(0)

    j = 1
    while 1:
        filename = path_dataset + newusername + '_' + str(newuserid) + '\\' + newusername + '_' + str(int(j / 10)) \
                   + '.jpg'
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetec.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Images Captured: {int(j / 10)}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 20), 2, cv2.LINE_AA)
            roi_gray = gray[y:y + h, x:x + w]
            # # # roi_color = frame[y:y+h, x:x+w]
            image_to_train = cv2.resize(src=roi_gray, dsize=(100, 100))
            if j % 10 == 0:
                if cv2.imwrite(filename, image_to_train):
                    j += 1
        cv2.imshow('Add new user', frame)
        if j == 500:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    return None


@app.route('/train')
def train():
    # from PIL import Image
    X = []
    y = []
    people = np.array(["obama_2", "trump_1", "putin_3", "unknown_0"])
    label_map = {label: num for num, label in enumerate(people)}
    path_dataset = "dataset\\"
    userlist = os.listdir(path_dataset)
    for person in userlist:
        filename = path_dataset + str(person)
        for imgname in os.listdir(filename):
            img = cv2.imread(path_dataset + str(person) + "\\" + imgname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(src=img, dsize=(100, 100))
            img = np.array(img)
            X.append(img)
            y.append(label_map[person])

    X1 = np.array(X)
    y1 = np.array(y)
    y2 = to_categorical(y1).astype(int)
    X = X1.reshape((200, 100, 100, 1))
    X2 = X / 255
    print(X2.shape, y2.shape)

    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3)
    Model = Sequential()
    shape = (100, 100, 1)
    Model.add(Conv2D(8, (4, 4), padding="valid"))
    Model.add(Activation("ReLU"))
    Model.add(Conv2D(16, (4, 4), padding="valid"))
    Model.add(Activation("ReLU"))
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Conv2D(32, (4, 4), padding="valid"))
    Model.add(Activation("ReLU"))
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Conv2D(64, (4, 4), padding="valid"))
    Model.add(Activation("ReLU"))
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Flatten())
    Model.add(Dense(64))
    Model.add(Activation("ReLU"))
    Model.add(Dense(4))
    Model.add(Activation("softmax"))
    Model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("start training")
    # Model.fit(X_train, y_train, batch_size=8, epochs=30)
    history = Model.fit(X_train, y_train, batch_size=8, epochs=4, validation_data=(X_test, y_test))
    pyplot.figure(figsize=(20, 10))
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

    y_hat = Model.predict(X_test)
    return None


if __name__ == '__main__':
    app.run(debug=True, port=8000)
