from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import applications
from datetime import datetime
import numpy as np
import cv2


def read_image(img_path, image_size):
    print('\nReading image...')
    original_image = cv2.imread(img_path)
    image = load_img(image_path, target_size=(image_size, image_size))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    return original_image, image


def predict_to_image(image, class_num, weights_path):
    print('Making prediction...')
    model = applications.InceptionV3(include_top=False, weights='imagenet')
    start = datetime.now()
    prediction = model.predict(image)

    model = Sequential()
    model.add(Conv2D(400, (3, 3), padding='same', input_shape=prediction.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(400, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(200, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(200, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(class_num, activation='sigmoid'))

    model.load_weights(weights_path)
    class_predicted = model.predict_classes(prediction)
    class_id = class_predicted[0]
    inv_map = {v: k for k, v in class_dictionary.items()}
    label = inv_map[class_id]
    proba = model.predict_proba(prediction)
    end = datetime.now()
    analysistime = round((end - start).total_seconds(), 2)
    acc = round((proba.max()) * 100, 2)
    return acc, label, proba, analysistime


if __name__ == "__main__":
    model_weights_path = 'modelweightsAcc1.0.h5'

    image_path = 'data/testimages/obj55.png'

    img_size = 400

    class_dictionary = {
      "obj1": 0,
      "obj2": 1,
      "obj3": 2,
      "obj4": 3,
      "obj5": 4
    }
    num_classes = class_dictionary.__len__()

    org_img, img = read_image(image_path, img_size)
    accuracy, img_class, probabilities, analysis_time = predict_to_image(img, num_classes, model_weights_path)

    print("\nAnalysis Time: {} ms".format(analysis_time))
    print("Probabilities: {}".format(probabilities))
    print("Class: {}".format(img_class))
    print("Accuracy: %{}".format(accuracy))

    cv2.putText(org_img, "Predicted: {}".format(img_class), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    cv2.putText(org_img, "Accuracy: %{}".format(accuracy), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    cv2.imshow("Predicted: {}".format(img_class) + " / Accuracy: %{}".format(accuracy), org_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
