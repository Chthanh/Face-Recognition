import cv2
from keras.models import load_model
import numpy as np
import pymongo
from bson.binary import Binary
import os
import matplotlib.pyplot as plt
from bson import ObjectId
import io
from PIL import Image
import base64

video = cv2.VideoCapture(0)
facedetec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

my_model = load_model('model.h5')
classname = np.array(["Obama", "Trump", "Putin","unknown"])


# conn = pymongo.MongoClient("mongodb://localhost:27017/")
# database = conn["Wisdom"]
# collections = database["FaceToTrainModel"]

# document = collections.find_one({"_id": ObjectId("64d34cedd725d7cd01c02923")})
# if document:
#     image_data= document["Image"]
#     image = Image.open(io.BytesIO(image_data))
#     image.show()
# else:
#     print("Image not found")


# path_dataset = "dataset\\"
# userlist = os.listdir(path_dataset)
# for person in userlist:
#     filename = path_dataset + person + "\\"
#     parse = person.split("_")
#     name = parse[0]
#     id = parse[1]
#
#     for imagename in os.listdir(filename):
#         img = cv2.imread(filename + imagename)
#         _, buffer = cv2.imencode(".jpg", img)
#         binary_image = buffer.tobytes()
#         image_data = {"Id": int(id), "Name": name,"Image" : binary_image}
#         #result = collections.insert_one(image_data)
#         if  imagename == "obama_0.jpg":
#             base64_data = base64.b64encode(binary_image).decode("utf-8")
#             image_stream = io.BytesIO(base64_data)
#             image = Image.open(image_stream)
#             plt.imshow(image)
#             plt.show()

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
           print(result)
           label1 = classname[np.argmax(result)]
           cv2.putText(frame, label1, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 2)

   cv2.imshow('Frame', frame)
   k = cv2.waitKey(1)
   if k == ord('q'):
       break
video.release()
cv2.destroyAllWindows()