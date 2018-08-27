from keras import layers, optimizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout, Add, AveragePooling2D
from keras.models import Model, Sequential, model_from_json

def conv(x, kernel, size, stride, activation=True, padding='same', **args):
    x = Conv2D(kernel, size, strides=stride, padding=padding, **args)(x)
    x = BatchNormalization(axis=2)(x)
    if activation:
        x = Activation('relu')(x)
    return x

def conv_block(x, kernels, size, stride, stage):
    k1, k2, k3 = kernels
    x_shortcut = x
    
    # residual is done once every 3 subblocks
    x = conv(x, k1, (1,1), (stride, stride), padding='valid', name='stage_{}_start'.format(stage))
    x = conv(x, k2, size, (1,1))
    x = conv(x, k3, (1,1), (1,1), False, padding='valid')
    x_shortcut = conv(x_shortcut, k3, (1,1), (stride, stride), False, padding='valid')
    # residual
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    return x
    
def id_block(x, kernels, size, stage):
    k1, k2, k3 = kernels
    x_shortcut = x
    
    x = conv(x, k1, (1,1), (1,1), padding='valid', name='stage_{}_start'.format(stage))
    x = conv(x, k2, (size, size), (1,1))
    x = conv(x, k3, (1,1), (1,1), False, padding='valid')
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    return x

def m_resnet(input_shape):
    x_input = Input(input_shape)

    x = Conv2D(32, (4,1), padding='same')(x_input)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    # stage 1
    x = conv_block(x, [32,32,128], 3, 1, 'a')
    x = id_block(x, [32,32,128], 3, 'b')
    x = id_block(x, [32,32,128], 3, 'c')

    # stage 2
    x = conv_block(x, [64,64,256], 3, 2, 'd')
    x = id_block(x, [64,64,256], 3, 'e')
    x = id_block(x, [64,64,256], 3, 'f')
    x = id_block(x, [64,64,256], 3, 'g')

    # stage 3
    x = conv_block(x, [128,128,512], 3, 2, 'h')
    x = id_block(x, [128,128,512], 3, 'i')
    x = id_block(x, [128,128,512], 3, 'j')
    x = id_block(x, [128,128,512], 3, 'k')
    x = id_block(x, [128,128,512], 3, 'l')
    x = id_block(x, [128,128,512], 3, 'm')
    
    # stage 4
    x = conv_block(x, [256,256,1024], 3, 2, 'n')
    x = id_block(x, [256,256,1024], 3, 'o')
    x = id_block(x, [256,256,1024], 3, 'p')

    x = AveragePooling2D((2,1))(x)
    x = Flatten()(x)
    x = Dense(7, activation='softmax')(x)

    model = Model(x_input, x, name='m_resnet')
    return model