import cv2
import numpy as np
import pandas as pd
import seaborn as sn
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot


# from PIL import Image
X = []
y = []
people = np.array(["obama_2", "trump_1", "putin_3","unknown_0"])
label_map = {label: num for num, label in enumerate(people)}
path_dataset = "dataset\\"
userlist = os.listdir(path_dataset)
for person in userlist:
    filename =  path_dataset + str(person)
    for imgname in os.listdir(filename):
        img = cv2.imread(filename+"\\"+imgname)
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
#Model.fit(X_train, y_train, batch_size=8, epochs=30)
history = Model.fit(X_train, y_train, batch_size=8, epochs=4, validation_data=(X_test,y_test))
pyplot.figure(figsize=(20,10))
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
y_pred = np.argmax(y_hat, axis=1)
y_test_label =  np.argmax(y_test, axis=1)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test_label, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test_label, y_pred, average='macro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test_label, y_pred, average='macro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test_label, y_pred, average='macro')
print('F1 score: %f' % f1)

auc = roc_auc_score(y_test, y_hat, multi_class='ovr')
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_test_label, y_pred)
print(matrix)
Model.save("model.h5")
