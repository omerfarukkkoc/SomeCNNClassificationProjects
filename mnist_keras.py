import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import numpy as np


#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("data/MNIST", one_hot=True)

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# print(x_test)

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)
#
# print(x_test)
# print(x_train[0])
# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()
#
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print("\nloss: ", val_loss)
print("acc: ", val_acc)

# model.save('mnist.model')

# new_model = keras.models.load_model('mnist.model')
predictions = model.predict(x_test)


print(np.argmax(predictions[0]))
print(np.argmax(predictions[1]))
print(np.argmax(predictions[2]))
print(np.argmax(predictions[3]))


f, subplot = plt.subplots(2, 2)
subplot[0, 0].imshow(x_test[0], cmap=plt.cm.binary)
subplot[0, 1].imshow(x_test[1], cmap=plt.cm.binary)
subplot[1, 0].imshow(x_test[2], cmap=plt.cm.binary)
subplot[1, 1].imshow(x_test[3], cmap=plt.cm.binary)

plt.show()

