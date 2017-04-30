from __future__ import print_function

import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.layers import Dropout
from configurations import *
from sklearn.externals import joblib
import argparse
from keras.callbacks import *
import sys
import theano
import theano.tensor as T
from keras import initializations
from keras.layers import BatchNormalization
import copy
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

'''
    DEFAULT CONFIGURATIONS
'''
def get_options():   
    
    parser = argparse.ArgumentParser(description='Short sample app')
    
    parser.add_argument('-out_dir', action="store", default='/scratch/cse/dual/cs5130287/Luna2016/output_final/',
                        dest="out_dir", type=str)
    
    parser.add_argument('-epochs', action="store", default=500, dest="epochs", type=int)
    
    parser.add_argument('-batch_size', action="store", default=2, dest="batch_size", type=int)    
    
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load_weights', action="store", default=False, dest="load_weights", type=bool)
    parser.add_argument('-filter_width', action="store", default=3, dest="filter_width",type=int)
    parser.add_argument('-stride', action="store", default=3, dest="stride",type=int)
    parser.add_argument('-model_file', action="store", default="", dest="model_file",type=str) #TODO
    parser.add_argument('-save_prefix', action="store", default="/scratch/cse/dual/cs5130287/Luna2016/goodModels/model_",
                        dest="save_prefix",type=str)
    opts = parser.parse_args(sys.argv[1:])    
        

    return opts



def dice_coef(y_true,y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    smooth = 0.
    intersection = K.sum(y_true*y_pred)
    
    
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)



def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def gaussian_init(shape, name=None, dim_ordering=None):
   return initializations.normal(shape, scale=0.001, name=name, dim_ordering=dim_ordering)

def get_unet_small(options):
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(32, 3, 3, activation='elu',border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='elu',border_mode='same', name='conv_1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool_1')(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Convolution2D(64, 3, 3, activation='elu',border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='elu',border_mode='same', name='conv_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool_2')(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Convolution2D(128, 3, 3, activation='elu',border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='elu',border_mode='same', name='conv_3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool_3')(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Convolution2D(256, 3, 3, activation='elu',border_mode='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(256, 3, 3, activation='elu',border_mode='same', name='conv_4')(conv4)
    conv4 = BatchNormalization()(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2), name='pool_4')(conv4)

    # conv5 = Convolution2D(512, 3, 3, activation='elu',border_mode='same')(pool4)
    # conv5 = Dropout(0.2)(conv5)
    # conv5 = Convolution2D(512, 3, 3, activation='elu',border_mode='same', name='conv_5')(conv5)

    # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    # conv6 = Convolution2D(256, 3, 3, activation='elu',border_mode='same')(up6)
    # conv6 = Dropout(0.2)(conv6)
    # conv6 = Convolution2D(256, 3, 3, activation='elu',border_mode='same', name='conv_6')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)

    conv7 = Convolution2D(128, 3, 3, activation='elu',border_mode='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(128, 3, 3, activation='elu',border_mode='same', name='conv_7')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='elu',border_mode='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(64, 3, 3, activation='elu',border_mode='same', name='conv_8')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='elu',border_mode='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(32, 3, 3, activation='elu',border_mode='same', name='conv_9')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', name='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.summary()
    model.compile(optimizer=Adam(lr=options.lr, clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef])

    return model



class WeightSave(Callback):
    def __init__(self, options):
        self.options = options

    def on_train_begin(self, logs={}):
        if self.options.load_weights:            
            print('LOADING WEIGHTS FROM : ' + self.options.model_file)
            weights = joblib.load( self.options.model_file )
            self.model.set_weights(weights)
    def on_epoch_end(self, epochs, logs = {}):
        cur_weights = self.model.get_weights()
        joblib.dump(cur_weights, self.options.save_prefix + '_script_on_epoch_' + str(epochs) + '_lr_' + str(self.options.lr) + '_WITH_STRIDES_' + str(self.options.stride) +'_FILTER_WIDTH_' + str(self.options.filter_width) + '.weights')

class Accuracy(Callback):
    def __init__(self,test_data_x,test_data_y):
        self.test_data_x=test_data_x
        self.test_data_y=test_data_y
        test = T.tensor4('test')
        pred = T.tensor4('pred')
        dc = dice_coef(test,pred)
        self.dc = theano.function([test,pred],dc)

    def on_epoch_end(self,epochs, logs = {}):
        predicted = self.model.predict(self.test_data_x)
        print ("Validation : %f"%self.dc(self.test_data_y,predicted))

def train(use_existing):
    print ("Yeh final hai yaar ")
    options = get_options()
    print ("epochs: %d"%options.epochs)
    print ("batch_size: %d"%options.batch_size)
    print ("filter_width: %d"%options.filter_width)
    print ("stride: %d"%options.stride)
    print ("learning rate: %f"%options.lr)
    sys.stdout.flush()

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train = np.load(options.out_dir+"trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(options.out_dir+"trainMasks.npy").astype(np.float32)

    # Renormalizing the masks
    imgs_mask_train[imgs_mask_train > 0.] = 1.0
    
    # Now the Test Data
    imgs_test = np.load(options.out_dir+"testImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load(options.out_dir+"testMasks.npy").astype(np.float32)
    # Renormalizing the test masks
    imgs_mask_test_true[imgs_mask_test_true > 0] = 1.0    

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet_small(options)
    weight_save = WeightSave(options)
    accuracy = Accuracy(copy.deepcopy(imgs_test),copy.deepcopy(imgs_mask_test_true))
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(x=imgs_train, y=imgs_mask_train, batch_size=options.batch_size, nb_epoch=options.epochs, verbose=1, shuffle=True
            ,callbacks=[weight_save, accuracy])
              # callbacks = [accuracy])
              # callbacks=[weight_save,accuracy])
    return model

if __name__ == '__main__':
    # print "epochs"
    model = train(False)
