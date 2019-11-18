from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from config import TrainConfig

def create_comma_model_large_dropout(row,col,ch, load_weights=False):  #change## parameter values // deepxplore Dave_dropout
    model = Sequential()

    model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode="same", input_shape=(row, col, ch)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())

    model.add(Dense(500))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(.25))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    if load_weights:
        model.load_weights('./Model.h5')

    print('Model is created and compiled..')
    return model


def my_train_generator():
    num_iters = X_train.shape[0] / batch_size
    num_iters = int(num_iters)
    print("num_iters: ", num_iters)
    while 1:

        for i in range(num_iters):
            #idx = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            idx = train_idx_shf[i*batch_size:(i+1)*batch_size]
            tmp = X_train[idx].astype('float32')
            tmp -= X_train_mean
            tmp /= 255.0
            yield tmp, y_train[idx]

def my_test_generator():
    num_iters = X_test.shape[0] / batch_size
    num_iters = int(num_iters)
    while 1:
        for i in range(num_iters):
            tmp = X_test[i*batch_size:(i+1)*batch_size].astype('float32')
            tmp -= X_train_mean
            tmp /= 255.0
            yield tmp, y_test[i*batch_size:(i+1)*batch_size]

if __name__ == "__main__":
    config = TrainConfig()

    ch = config.num_channels
    row = config.img_height
    col = config.img_width
    num_epoch = config.num_epoch
    batch_size = config.batch_size
    data_path = config.data_path
    
    print( "Loading training data...")
    # print( "Data path: " + data_path + "/X_train_round2_" + config.data_name + ".npy")

    X_train1 = np.load(data_path + "/X_train_round2_" + config.data_name + "_01-03--13-46-00.npy")
    y_train1 = np.load(data_path + "/y_train_round2_" + config.data_name + "_01-03--13-46-00.npy")
    X_train2 = np.load(data_path + "/X_train_round2_" + config.data_name + "_01-03--11-24-5.npy")
    y_train2 = np.load(data_path + "/y_train_round2_" + config.data_name + "_01-03--11-24-5.npy")
    X_train3 = np.load(data_path + "/X_train_round2_" + config.data_name + "_02-02--10-16-58.npy")
    y_train3 = np.load(data_path + "/y_train_round2_" + config.data_name + "_02-02--10-16-58.npy")
    X_train4 = np.load(data_path + "/X_train_round2_" + config.data_name + "_05-12--22-20-00.npy")
    y_train4 = np.load(data_path + "/y_train_round2_" + config.data_name + "_05-12--22-20-00.npy")
    X_train5 = np.load(data_path + "/X_train_round2_" + config.data_name + "_06-02--21-39-29.npy")
    y_train5 = np.load(data_path + "/y_train_round2_" + config.data_name + "_06-02--21-39-29.npy")
    # X_train6 = np.load(
    #     data_path + "/X_train_round2_" + config.data_name + "_80m160_06-02--21-39-29_123961to223791_f10.npy")
    # y_train6 = np.load(
    #     data_path + "/y_train_round2_" + config.data_name + "_80m160_06-02--21-39-29_123961to223791_f10.npy")


    # use last frames from all parts as validation set
    if config.val_part == 6:
         # X_train = X_train1[:13267]
         # y_train = y_train1[:13267]
         # X_test = X_train1[13267:]
         # y_test = y_train1[13267:]
         X_train = np.concatenate((X_train1[:13267], X_train2[:9940], X_train3[:14280], X_train4[:11402], X_train5[:6989]),
                             axis=0)
         y_train = np.concatenate((y_train1[:13267], y_train2[:9940], y_train3[:14280], y_train4[:11402], y_train5[:6989]),
                             axis=0)
         X_test = np.concatenate((X_train1[13267:], X_train2[9940:], X_train3[14280:], X_train4[11402:], X_train5[6989:]),
                            axis=0)
         y_test = np.concatenate((y_train1[13267:], y_train2[9940:], y_train3[14280:], y_train4[11402:], y_train5[6989:]),
                            axis=0)

    print( "X_train shape:" + str(X_train.shape))
    print( "X_test shape:" + str(X_test.shape))
    print( "y_train shape:" + str(y_train.shape))
    print( "y_test shape:" + str(y_test.shape))
    
    np.random.seed(1235)
    train_idx_shf = np.random.permutation(X_train.shape[0])
    
    print( "Computing training set mean...")
    X_train_mean = np.mean(X_train, axis=0, keepdims=True)
    
    print( "Saving training set mean...")
    np.save(config.X_train_mean_path, X_train_mean)
    
    print( "Creating model...")

    if config.model_name == "comma_large_dropout":
        model = create_comma_model_large_dropout(row,col,ch)
        save_model_name = './Model.h5'

        
    print( model.summary())

    # checkpoint
    path = "./models"
    filepath = path + "/weights_" + config.data_name + "_" + config.model_name + "-{epoch:02d}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    
    iters_train = X_train.shape[0]
    iters_train -= iters_train % batch_size
    iters_test = X_test.shape[0]
    iters_test -= iters_test % batch_size
    
    model.fit_generator(my_train_generator(),
        nb_epoch=num_epoch,
        samples_per_epoch=iters_train,
        validation_data=my_test_generator(),
        nb_val_samples=iters_test,
        callbacks=callbacks_list,
        nb_worker=1
    )

    # save model
    model.save_weights(save_model_name)