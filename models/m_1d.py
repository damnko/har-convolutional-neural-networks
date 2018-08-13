from keras import layers, optimizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Dropout, ELU
from keras.models import Model, Sequential, model_from_json

# simple model which convolves only with depth 1, has 2
# conv layers and 1 dense
def m_1d(input_shape):
    X_input = Input(input_shape)
    
    """
    if I'm using channel_last input_shape is something like
    [12000, 128, 6, 1] where 12000 is the nr of samples
    and X_input shape is like
    [128, 6, 1]
    """

    X = Conv1D(30, 5, name='conv0')(X_input)
    # in channel_last normalization is done on axis=1
    # which is the time domain
    X = BatchNormalization(axis = 1, name='bn0')(X)
    X = Activation('relu')(X)
    # X = ELU()(X)
    
    X = Dropout(0.3)(X)
    X = MaxPooling1D(2, name='max_pool0')(X)
    
    X = Conv1D(40, 5, name='conv1')(X)
    X = BatchNormalization(axis = 1, name='bn1')(X)
    X = Activation('relu')(X)
    
    X = Dropout(0.3)(X)
    X = MaxPooling1D(2, name='max_pool1')(X)
    
    X = Flatten()(X)
    #X = Dense(150, activation=K.elu, name='fc')(X)
    X = Dense(150, activation="relu", name='fc')(X)
    X = Dropout(0.2)(X)
    X = Dense(7, activation='softmax', name='softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='m_1d')    
    
    return model