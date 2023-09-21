import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

path_dataset = "dataset\\"

username = 'unknown'
#username = 'obama'
#username = "putin"

userid = 0
userfolder = "dataset\\" + username + "_" + str(userid)
if not os.path.isdir(userfolder):
    os.makedirs(userfolder)

cap = cv2.VideoCapture(0)
#facedetec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

j = 1
while 1:
    filename = 'dataset\\'+username+'_'+str(userid)+'\\'+ username + '_' + str(int(j/10)) + '.jpg'
    # print(filename)
    #frame = cv2.imread(filename)
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Images Captured: {int(j/10)}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,cv2.LINE_AA)
        roi_gray = gray[y:y + h, x:x + w]
        # # # roi_color = frame[y:y+h, x:x+w]
        image_to_train = cv2.resize(src=roi_gray, dsize=(100, 100))
        if cv2.imwrite(filename, image_to_train):
            j += 1
    cv2.imshow('Add new user',frame)
    if j ==500:
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()