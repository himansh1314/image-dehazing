# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, LeakyReLU, Dropout, Input, Concatenate, Activation, BatchNormalization, ReLU, AveragePooling2D
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2

def Layer(previous_layer, filters):
    """ As described in the fully convolutional densenet paper, this function defines a LAYER
    that would be used in DenseBlock. Takes input the previous layer and number of filters and
    and returns LAYER. """
    next_layer = BatchNormalization(momentum = 0.9, epsilon = 1.1e-5)(previous_layer)
    next_layer = ReLU()(next_layer)
    next_layer = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal', activation = 'linear')(next_layer)
    next_layer = Dropout(0.2)(next_layer)
    return next_layer

def TransitionDown(previous_layer, filters):
    """Transition Down layer as described in FC Densenet paper"""
    next_layer = BatchNormalization(momentum = 0.9, epsilon = 1.1e-5)(previous_layer)
    next_layer = ReLU()(next_layer)
    next_layer = Conv2D(filters = filters, kernel_size = 1, strides = 1, kernel_initializer = 'he_normal', padding = 'same', activation = 'linear', kernel_regularizer = l2(1e-4))(next_layer)
    next_layer = Dropout(0.2)(next_layer)
    next_layer = MaxPooling2D()(next_layer)
    return next_layer

def TransitionUp(previous_layer, filters):
    """ Defines a TransitionUp layer and uses Conv2DTranspose. Takes input previous layers 
    and number of filters"""
    next_layer = Conv2DTranspose(filters = filters, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu', kernel_regularizer = l2(1e-4))(previous_layer)
    return next_layer

def DenseBlock(previous_layer, growth_rate, num_layers = 4):
    layers_list = [previous_layer]
    for i in range(0,num_layers):
        next_layer = Layer(previous_layer, growth_rate)
        layers_list.append(next_layer)
        previous_layer = Concatenate()([previous_layer, next_layer])
    return previous_layer



def DenseNet():
    inputs  = Input([256,256,3])
    conv = Conv2D(48, strides = 1, kernel_size = 3, padding = 'same')(inputs)
    conv = BatchNormalization(epsilon = 1.1e-5, momentum = 0.9)(conv)
    conv = ReLU()(conv)
    
    DB1 = DenseBlock(conv, 12)
    TD1 = TransitionDown(DB1, 48)
    
    DB2 = DenseBlock(TD1, 12)
    TD2 = TransitionDown(DB2, 48)
    
    DB3 = DenseBlock(TD2, 12)
    TD3 = TransitionDown(DB3, 48)
    
    DB4 = DenseBlock(TD3, 12)
    TD4 = TransitionDown(DB4, 48)
    
    DB5 = DenseBlock(TD4, 12)
    TD5 = TransitionDown(DB5, 48)
    
    DB_BottleNeck = DenseBlock(TD5, 15)
    
    TU1 = TransitionUp(DB_BottleNeck, 48)
    concat = Concatenate()([TU1, DB5])
    DBD5 = DenseBlock(concat, 12)
    
    TU2 = TransitionUp(DBD5, 48)
    concat = Concatenate()([TU2, DB4])
    DBD4 = DenseBlock(concat, 12)
    
    TU3 = TransitionUp(DBD4, 48)
    concat = Concatenate()([TU3, DB3])
    DBD3 = DenseBlock(concat, 12)
    
    TU4 = TransitionUp(DBD3, 48)
    concat = Concatenate()([TU4, DB2])
    DBD2 = DenseBlock(concat, 12)
    
    TU5 = TransitionUp(DBD2, 48)
    concat = Concatenate()([TU5, DB1])
    DBD1 = DenseBlock(concat, 12)
    
    output = Conv2D(filters = 3, kernel_size = 1, strides = 1, padding = 'same', kernel_initializer = 'he_normal', activation = 'tanh')(DBD1)
    model = Model(inputs = inputs, outputs = output)
    model.summary()
    plot_model(model)
    return model
# from densenet_keras import create_tiramisu
# model = create_tiramisu(3, nb_layers_per_block=4, p=0.2, wd=1e-4)
# model.summary()
# plot_model(model, show_shapes = True)