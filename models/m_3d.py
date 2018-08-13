from keras import layers, optimizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Dropout, ELU
from keras.models import Model, Sequential, model_from_json

# model with both 1d and 3d convolutions to cross correlate x,y,z sensor values
# has 3 conv layers and 1 dense
def m_3d(input_shape):
    X_input = Input(input_shape)
    
    # if I'm using channel_first input_shape is something like
    # [12000, 1, 128, 6] where 12000 is the nr of samples
    # and X_input shape is like
    # [1, 128, 6]
    
    X = Conv2D(20, (4,2), name='conv0')(X_input)
    # if using channel_first, normalization has to be along
    # axis=2, which is the time domain (128)
    X = BatchNormalization(axis = 2, name='bn0')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(40, (2,2), name='conv1')(X)
    X = BatchNormalization(axis = 2, name='bn1')(X)
    X = Activation('relu')(X)
    
    X = Dropout(0.2)(X)
    
    #X = MaxPooling2D(name='max_pool1')(X)
    
    X = Conv2D(60, (4,4), name='conv2', padding='same')(X)
    X = BatchNormalization(axis = 2, name='bn2')(X)
    X = Activation('relu')(X)
    
    X = Dropout(0.3)(X)
    
    X = MaxPooling2D(name='max_pool1')(X)
    
    X = Flatten()(X)
    X = Dense(150, activation='relu', name='fc')(X)
    X = Dropout(0.3)(X)
    X = Dense(7, activation='softmax', name='softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='m_3d')
    
    return model