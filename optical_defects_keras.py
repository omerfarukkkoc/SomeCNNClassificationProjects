import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import random
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD



DATADIR = "D:\\AIWorks\\0.Datasets\\optical_defects\\optical_defects"

CATEGORIES = ["Blur", "Crack", "Scratch"]


IMG_SIZE = 250

classes_num = 3


training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()

print("\nTraining Data length: ", len(training_data))


random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

# print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# pickle_out = open("BlurAndCrackX.pickle","wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
#
# pickle_out = open("BlurAndCrackY.pickle","wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()
#
# pickle_in = open("BlurAndCrackX.pickle","rb")
# X = pickle.load(pickle_in)
#
# pickle_in = open("BlurAndCrackY.pickle","rb")
# y = pickle.load(pickle_in)

X = X/255.0

# print(len(X))


# print(X.shape[1:])

#
# def cnn_model():
#     model = Sequential()
#
#     model.add(Conv2D(50, (3, 3), padding='same', input_shape=X.shape[1:], activation='relu'))
#     model.add(Conv2D(50, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))
#
#     model.add(Conv2D(100, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(100, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))
#
#     model.add(Conv2D(200, (3, 3), padding='same',activation='relu'))
#     model.add(Conv2D(200, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))
#
#     model.add(Flatten())
#     model.add(Dense(512,  activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='softmax'))
#     return model
#
#
#
# model = cnn_model()
#
#
# lr = 0.01
# sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='binary_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
#
#
# model.fit(X, y, batch_size=64, epochs=10, validation_split=0.25)
#
# model.save('BlurAndCrack.model')




# model = Sequential()
# model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.50))
#
# model.add(Conv2D(128, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))
#
#
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.12))
#
# model.add(Flatten())
# model.add(Dense(32,  activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(1, activation='softmax'))


model = Sequential()
#
# model.add(Conv2D(40, kernel_size=5, padding="same",input_shape=X.shape[1:], activation= 'relu'))
# model.add(Conv2D(50, kernel_size=5, padding="valid", activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Activation("relu"))
# model.add(Dropout(0.2))
#
# model.add(Dense(1))
# model.add(Activation("softmax"))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.add(Conv2D(32, (3, 3), padding="same", input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epoch = 2
H =model.fit(X, y, batch_size=8, epochs=epoch, validation_split=0.25)

model.save('optical_defects.model')
print("\nModel saved")

plt.style.use("ggplot")
plt.figure()
N = epoch
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
# plt.savefig(args["plot"])



#
#
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# # model.compile(loss='binary_crossentropy',
# #               optimizer='rmsprop',
# #               metrics=['accuracy'])
#
#
# opt = keras.optimizers.Adam(lr=0.00001, decay=1e-6)
#
#
# # opt = keras.optimizers.adadelta(lr=0.001, decay=1e-6)
#
# model.compile(loss='binary_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])
#
#
# model.fit(X, y, batch_size=64, epochs=10, validation_split=0.4)
#
# model.save('BlurAndCrack.model')
#

