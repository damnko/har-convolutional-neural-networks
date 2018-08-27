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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import layers, optimizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Dropout, ELU, Conv2DTranspose
from keras.models import Model, Sequential, model_from_json
from keras.optimizers import SGD

sn.set(style="white", context="talk")
sn.set(style="whitegrid")

from utils import terminate, outdir, load_dataset, show_stats, ohe_to_label, conf_matrix, export_model, f1
from models import m_1d, m_1d2d, m_1d2d_01, m_3d

from keras.layers import UpSampling2D
K.set_image_data_format('channels_first')

general_conf = {
    'debug': False,
    'prod': True,
    'export_models': True,
    'batch_size': 32,
    'iterations': 1,
    'datasets': [
        '',
        #'-augmented',
        #'-with-trans'
    ]
}

X_train, X_test, Y_train, Y_test = load_dataset('')

X_train_noise = X_train + np.random.normal(loc=0, scale=0.15, size=X_train.shape)
X_test_noise = X_test + np.random.normal(loc=0, scale=0.15, size=X_test.shape)

# channel first
X_train_4ch = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test_4ch = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

X_train_noise_4ch = X_train_noise.reshape(X_train_noise.shape[0], 1, X_train_noise.shape[1], X_train_noise.shape[2])
X_test_noise_4ch = X_test_noise.reshape(X_test_noise.shape[0], 1, X_test_noise.shape[1], X_test_noise.shape[2])

input_shape = [X_train_4ch.shape[1], X_train_4ch.shape[2], X_train_4ch.shape[3]]

X_input = Input(input_shape)
"""
#x = BatchNormalization(axis = 2, name='bn')(X_input)
x = Conv2D(32, (4,1), activation='relu', padding='same', name='enc0')(X_input)
#x = BatchNormalization(axis = 2, name='bn1')(x)
x = Conv2D(64, (3,2), activation='relu', padding='same', name='enc1')(x)
x = MaxPooling2D(name='encoded')(x)

x = Conv2D(64, (3,2), activation='relu', padding='same', name='dec0')(x)
x = UpSampling2D(name='up1')(x)
#x = BatchNormalization(axis = 2, name='bn2')(x)
x = Conv2D(32, (4,1), activation='relu', padding='same', name='dec2')(x)
#x = BatchNormalization(axis = 2, name='bn3')(x)
x = Conv2D(1, (3,3), name='decoded', padding='same')(x)
"""

# long version is

x = Conv2D(32, (5,1), activation='relu')(X_input)
x = Conv2D(64, (7,1), activation='relu')(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D(name='encoded1')(x)
ae1_enc_shape = x.shape.as_list()

x = UpSampling2D()(x)
x = Conv2DTranspose(64, (3,3), activation='relu')(x)
x = Conv2DTranspose(32, (7,1), activation='relu')(x)
x = Conv2DTranspose(1, (5,1))(x)


# short version is
"""
x = Conv2D(32, (5,1), activation='relu')(X_input)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(name='encoded1')(x)
ae1_enc_shape = x.shape.as_list()

x = UpSampling2D()(x)
x = Conv2DTranspose(32, (3,3), activation='relu')(x)
x = Conv2DTranspose(1, (5,1))(x)
"""

# the original was
"""
X_input = Input(input_shape)
x = Conv2D(64, (4,1), activation='relu', padding='same')(X_input)
x = Conv2D(32, (3,2), activation='relu', padding='same')(x)
x = MaxPooling2D(name='encoded1')(x)
ae1_enc_shape = x.shape.as_list()

x = Conv2D(32, (3,2), activation='relu', padding='same')(x)
x = UpSampling2D()(x)
x = Conv2D(64, (4,1), activation='relu', padding='same')(x)
x = Conv2D(1, (3,3), padding='same')(x)
"""


model = Model(input=X_input, output=x, name='conv_ae')
model.compile(loss='mse', optimizer='rmsprop')

model.summary()


# train the model
history = model.fit(x = X_train_noise_4ch, y = X_train_4ch,
                    epochs=200,
                    batch_size=64,
                    callbacks=[
                        EarlyStopping(patience=15, monitor='val_loss', min_delta=0, mode='min'),
                        ModelCheckpoint('best-weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
                    ],
                    validation_data=(X_test_noise_4ch, X_test_4ch))

model.load_weights('best-weights.h5')
prediction = model.predict(X_test_noise_4ch)

ints = [139,148,155,215,226,305,317,320,335,379,394,435,449,469,579,601,621,704,707,739,833,856,883,891,994]
for i in ints:
    prediction_res = prediction[i, 0,:,0]
    # plot original and reconstructed
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 4))
    x = np.arange(len(X_test_4ch[i, 0, :, 0]))
    sn.lineplot(x, X_test_noise_4ch[i, 0, :, 0], color='red', alpha=0.7, label='Noisy', ax=ax1)
    sn.lineplot(x, X_test_4ch[i, 0, :, 0], color='red', label='Test', ax=ax1)
    sn.lineplot(x, prediction_res, color='green', label='Rebuilt', ax=ax1)
    ax1.lines[0].set_linestyle("--")
    ax1.legend(loc='upper right')
    fig.savefig('{}.png'.format(i))
    fig.savefig('{}.eps'.format(i), format='eps', dpi=1000)