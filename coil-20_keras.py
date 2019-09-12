from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras import applications
import numpy as np
import math


def create_training_data(data_directory, image_size, batch_size, dataset_name, model):
    print("Creating the {} dataset...".format(dataset_name))
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        data_directory,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    nb_train_samples = len(generator.filenames)
    predict_size = int(math.ceil(nb_train_samples / batch_size))
    features = model.predict_generator(generator, predict_size)
    np.save(dataset_name, features)
    print("{} saved".format(dataset_name))


def read_training_data(data_directory, image_size, batch_size, dataset_name):
    print("Reading the {} dataset...".format(dataset_name))
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        data_directory,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    class_num = len(generator_top.class_indices)
    datasett = np.load(dataset_name)
    labelss = generator_top.classes
    labelss = to_categorical(labelss, num_classes=class_num)
    return datasett, labelss, class_num


def create_cnn_model(dataset_shape, class_num, optimizer):
    model = Sequential()

    model.add(Conv2D(400, (3, 3), padding='same', input_shape=dataset_shape))
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

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def train(x, y, model, epoch, batch_size, validation_split, weights_path):
    # early_stopping = EarlyStopping(monitor='val_loss',
    #                                min_delta=0,
    #                                patience=0,
    #                                verbose=0,
    #                                mode='auto')
    hist = model.fit(x, y,
                        epochs=epoch,
                        batch_size=batch_size,
                        validation_split=validation_split)
    model.save_weights(weights_path)
    # model.save('SurfaceModel.model')
    model.summary()
    return hist


if __name__ == "__main__":
    img_size = 400
    # data_dir = 'data/coil-20'
    data_dir = 'D:\\AIWorks\\0.Datasets\\COIL-20\\coil-20'
    datasetName = 'coil20_dataset.npy'
    batch = 4
    lr = 1e-3
    epochs = 5
    val_split = 0.10
    model_weights_path = 'modelweights.h5'

    inception_model = applications.InceptionV3(include_top=False, weights='imagenet')

    sgd_optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    # create_training_data(data_dir, img_size, batch, datasetName, inception_model)
    dataset, labels, num_classes = read_training_data(data_dir, img_size, batch, datasetName)
    sequential_model = create_cnn_model(dataset.shape[1:], num_classes, optimizer=sgd_optimizer)
    history = train(dataset, labels, sequential_model, epochs, batch, val_split, model_weights_path)

    print("Image Dimensions:", img_size, "x", img_size)
    print("Num Classes: ", num_classes)
    print("Batch Size: {}".format(batch))
    print("Epoch: {}".format(epochs))
    print("Learning Rate: {}".format(lr))
    print("Val Split: {}".format(val_split))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Acc and Loss Graph.png")
    plt.show()
