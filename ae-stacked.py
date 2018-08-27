# generic imports
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sn
import h5py
import time
from subprocess import call
import telegram_send
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import layers, optimizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Dropout, ELU, UpSampling2D, Conv2DTranspose
from keras.models import Model, Sequential, model_from_json

sn.set(style="white", context="talk")

from utils import terminate, outdir, load_dataset, show_stats, ohe_to_label, conf_matrix, export_model, f1
from models import m_1d, m_1d2d, m_1d2d_01, m_3d

# needed to work on GPU
K.set_image_data_format('channels_first')

def gen_noise(shape, x, zero_data = False, scale=0.15):
    if not zero_data:
        return np.random.normal(loc=0, scale=scale, size=shape)
    # zero out some values
    zeros = np.zeros(shape)
    unif_n = np.arange(x.size)/(x.size-1)
    np.random.shuffle(unif_n)
    # percentage of frames to keep, 1-keep will be set to zero
    keep = 0.97
    mask = (unif_n>keep).reshape(shape)
    x[mask] = zeros[mask]
    # -x will zero out 1-keep % of elements
    return -x

def callbacks(name, tensorboard = False):
    callbacks = [
        ModelCheckpoint('weights-{}.h5'.format(str(name)), monitor='val_loss', save_best_only=True, save_weights_only=True),
        EarlyStopping(patience=35, monitor='val_loss', min_delta=0, mode='min')
    ]
    if tensorboard:
        callbacks.append(TensorBoard(log_dir='./logs/{}'.format(name), histogram_freq=0, write_graph=True, write_images=True))
    return callbacks

class_conversion = {
    '0': 'falling',
    '1': 'jumping',
    '2': 'lying',
    '3': 'running',
    '4': 'sitting',
    '5': 'standing',
    '6': 'walking'
}

general_conf = {
    'model_name': 'ae-stacked-long-gaus',
    'debug': False,
    'prod': True,
    'export_models': True,
    'zero_type_noise': False,
    'batch_size': 64,
    'iterations': 400,
    'datasets': [
        #'',
        '-augmented',
        #'-with-trans'
    ]
}

X_train, X_test, Y_train, Y_test = load_dataset('')

X_train_noise = X_train + gen_noise(X_train.shape, X_train, general_conf['zero_type_noise'])
X_test_noise = X_test + gen_noise(X_test.shape, X_test, general_conf['zero_type_noise'])

# channel first reshaping
X_train_4ch = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test_4ch = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
X_train_noise_4ch = X_train_noise.reshape(X_train_noise.shape[0], 1, X_train_noise.shape[1], X_train_noise.shape[2])
X_test_noise_4ch = X_test_noise.reshape(X_test_noise.shape[0], 1, X_test_noise.shape[1], X_test_noise.shape[2])

input_shape = [X_train_4ch.shape[1], X_train_4ch.shape[2], X_train_4ch.shape[3]]


###########################
#### FIRST AUTOENCODER ####

X_input = Input(input_shape)
x = Conv2D(32, (5,1), activation='relu')(X_input)
x = Conv2D(64, (7,1), activation='relu')(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D(name='encoded1')(x)
ae1_enc_shape = x.shape.as_list()
print(ae1_enc_shape)
x = UpSampling2D()(x)
x = Conv2DTranspose(64, (3,3), activation='relu')(x)
x = Conv2DTranspose(32, (7,1), activation='relu')(x)
x = Conv2DTranspose(1, (5,1))(x)

ae1 = Model(input=X_input, output=x, name='ae1')
ae1.compile(loss='mse', optimizer='rmsprop')

ae1.summary()

# train the model, if not already trained
if not Path("weights-ae1-long-gaus.h5").is_file():
    history = ae1.fit(x = X_train_noise_4ch, y = X_train_4ch,
                        epochs=general_conf['iterations'],
                        batch_size=general_conf['batch_size'],
                        callbacks=callbacks('ae1-long-gaus', True),
                        validation_data=(X_test_noise_4ch, X_test_4ch))

# load best weights
ae1.load_weights('weights-ae1-long-gaus.h5')


# get the output of the encoded layer
X_input = Input(input_shape)
enc1_layer = ae1.get_layer('encoded1')
ae1_encoder = Model(ae1.input, enc1_layer.output)
ae1_encoder.compile(loss='mse', optimizer='rmsprop')


"""
# I could have expanded the model starting from the encoder part of ae1 by doing
x = MaxPooling2D()(enc1_layer.output)
# this can be also used to remove layers from the model
model.layers.pop()
"""

############################
#### SECOND AUTOENCODER ####

# this is the input of the second autoencoder
ae2_input = ae1_encoder.predict(X_train_4ch)

# input shape is the output of the encoder part of ae1
X_input1 = Input(ae1_enc_shape[1:])
x1 = Conv2D(256, (3,3), activation='relu', padding='same')(X_input1)
x1 = Conv2D(512, (2,2), activation='relu', padding='same')(x1)
x1 = MaxPooling2D((2,2), name='encoded2')(x1)

x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(256, (2,2), activation='relu', padding='same')(x1)
x1 = Conv2D(128, (3,3), activation='relu', padding='same')(x1)

ae2 = Model(X_input1, x1, name='ae2')
ae2.compile(loss='mse', optimizer='rmsprop')
print(ae2.summary())

# split train and test
qnt_train = round(ae2_input.shape[0] * 0.8)
train = ae2_input[:qnt_train]
train_noise = train + gen_noise(train.shape, train, general_conf['zero_type_noise'])
test = ae2_input[qnt_train:]
test_noise = test + gen_noise(test.shape, test, general_conf['zero_type_noise'])

# train should be added with some noise
if not Path("weights-ae2-long-gaus.h5").is_file():
    history = ae2.fit(x = train_noise, y = train,
                        epochs=general_conf['iterations'],
                        batch_size=general_conf['batch_size'],
                        callbacks=callbacks('ae2-long-gaus', True),
                        validation_data=(test_noise, test))

ae2.load_weights('weights-ae2-long-gaus.h5')


######################
#### STACKING AES ####

enc_layer_ae1 = ae1.get_layer('encoded1')
enc_layer_ae2 = ae2.get_layer('encoded2')

enc_layer_ae2 = enc_layer_ae2(enc_layer_ae1.output)
full_output = Flatten()(enc_layer_ae2)
full_output = Dense(150, activation='relu')(full_output)
full_output = Dense(7, activation='softmax')(full_output)
full_model = Model(ae1.input, full_output)
full_model.summary()

# freeze the layers of the first 2 stacked autoencoders
for layer in full_model.layers[:5]:
    layer.trainable = False

full_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', f1])

if not Path("weights-ae-stacked-long-gaus.h5").is_file():
    full_model.fit(x = X_train_4ch, y = Y_train,
                        epochs=general_conf['iterations'],
                        batch_size=general_conf['batch_size'],
                        callbacks=callbacks('ae-stacked-long-gaus', True),
                        validation_data=(X_test_4ch, Y_test))

full_model.load_weights('weights-ae-stacked-long-gaus.h5')

##########################
#### CHECKING RESULTS ####

preds = full_model.evaluate(x = X_test_4ch, y = Y_test)

start_time = time.time()
show_stats(start_time, preds)

# output results
predictions = full_model.predict(X_test_4ch)
Y_pred = ohe_to_label(predictions)
Y_true = ohe_to_label(Y_test)

conf_matrix(Y_true, Y_pred, class_conversion, general_conf['model_name'], save = True)

export_model(full_model, general_conf['model_name'])
