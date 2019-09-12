import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import numpy as np
import cv2
import os


from keras import applications
from keras.utils.np_utils import to_categorical

DATADIR = "D:\\AIWorks\\0.Datasets\\optical_defects\\optical_defects"

CATEGORIES = ["Blur", "Crack"]


IMG_SIZE = 250


model = keras.models.load_model("optical_defects.model")
# model = keras.models.load_model("BlurAndCrack.model")

testImagePath = os.path.join(DATADIR, "TestImages/crack1.jpg")

inputimage = cv2.imread(testImagePath)

grayImg = cv2.cvtColor(inputimage, cv2.COLOR_RGB2GRAY)
resizeImg = cv2.resize(grayImg, (IMG_SIZE, IMG_SIZE))
grayAndScaleImg = resizeImg.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

prediction = model.predict(grayAndScaleImg)

# print("\n"+prediction)


# (blur, crack) = model.predict(grayAndScaleImg)
#
# proba = blur if blur > crack else crack
# label = "{}: {:.2f}%".format(label, proba * 100)
#
# print(proba)

predictionResultText = "Prediction Result: " + CATEGORIES[int(prediction[0][0])]

print("\n\n"+predictionResultText)

cv2.putText(inputimage, predictionResultText, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow(predictionResultText, inputimage)

#
# cv2.imshow("gray", grayImg)
# cv2.imshow("resize", resizeImg)
# # cv2.imshow("grayandscale", grayAndScaleImg)

cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow("Predictions",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#
#print(np.argmax(predictions[0]))
# print(np.argmax(predictions[1]))
# print(np.argmax(predictions[2]))
# print(np.argmax(predictions[3]))
#
#

# f, subplot = plt.subplots(2, 2)
# subplot[0, 0].imshow(img_array, cmap=plt.cm.binary)
# subplot[0, 1].imshow(img_array, cmap=plt.cm.binary)
# subplot[1, 0].imshow(img_array, cmap=plt.cm.binary)
# subplot[1, 1].imshow(img_array, cmap=plt.cm.binary)
#
# plt.show()
#
