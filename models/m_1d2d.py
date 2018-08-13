from keras import layers, optimizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Dropout, ELU
from keras.models import Model, Sequential, model_from_json

def conv_batch_layer(X, depth, kernel_size, id, **args):
    X = Conv2D(depth, kernel_size, name='conv{}'.format(str(id)), **args)(X)
    X = BatchNormalization(axis = 2, name='bn{}'.format(str(id)))(X)
    X = Activation('relu')(X)
    return X

def m_1d2d(input_shape):
    X_input = Input(input_shape)
    
    X = conv_batch_layer(X_input, 16, (4,1), 0)
    X = conv_batch_layer(X, 32, (4,1), 1)
    X = Dropout(0.3)(X)
    X = conv_batch_layer(X, 64, (3,1), 2)
    X = conv_batch_layer(X, 64, (3,3), 3, padding='same')
    X = Dropout(0.3)(X)
    X = MaxPooling2D(name='max_pool0')(X)
    X = conv_batch_layer(X, 64, (2,3), 4, padding='same')
    X = MaxPooling2D(name='max_pool1')(X)
    X = conv_batch_layer(X, 64, (2,3), 5, padding='same')
    #X = MaxPooling2D(name='max_pool2')(X)
    X = Dropout(0.3)(X)
    X = Flatten()(X)
    X = Dense(150, activation='relu', name='fc')(X)
    X = Dropout(0.2)(X)
    X = Dense(7, activation='softmax', name='softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='m_1d2d')    
    
    return model