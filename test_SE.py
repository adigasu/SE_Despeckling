"Shared Encoder based Denoising of OCT Images"

import os
import glob
import numpy as np
from PIL import Image

import keras as keras
from keras.models import Model
from keras.layers import Activation
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras import backend as K
import keras_contrib

K.set_image_data_format('channels_last')
print keras.backend.image_data_format()

############################ Parameters ##################################
modelName = 'SSE1'                  # model name: SE or SSE1
datapath = './test_images/'         # Test image path

ipDepth = 1                         # Input image depth
ae_f = 64                           # Autoencoder Initial filter size

lrate = 0.1                         # Learning Rate
decay_A1 = 2e-5                     # Weight decay of autoencoder 1 (A1)
decay_A2 = 6e-5                     # Weight decay of autoencoder 2 (A2)
batch_sz = 8                        # Batch size

##############################################################
# loss function
def my_loss(y_true, y_pred):
    l1_loss = K.mean(K.abs(y_pred - y_true))

    return l1_loss

################################### Network #############################################
# Shared Encoder network
def conv2d(layer_input, filters, f_size=5):
    lay = Conv2D(filters, kernel_size=f_size)(layer_input)
    lay = BatchNormalization()(lay)
    lay_1 = Activation('relu')(lay)

    lay = Dropout(0.2)(lay_1)

    lay = Conv2D(filters, (f_size, f_size))(lay)
    lay = BatchNormalization()(lay)
    lay_2 = Activation('relu')(lay)

    lay = concatenate([Cropping2D(cropping=(2, 2))(lay_1), lay_2], axis=-1)

    return lay

# Encoder network
def getSharedEncoder(H, W, nCh):

    inputs = Input((H, W, nCh))
    # print('Input:', inputs._keras_shape)

    e1 = conv2d(inputs, ae_f)
    p1 = MaxPooling2D(pool_size=(2, 2))(e1)
    #
    e2 = conv2d(p1, ae_f*2)
    p2 = MaxPooling2D(pool_size=(2, 2))(e2)
    #
    e3 = conv2d(p2, ae_f*4)
    return inputs, e3

# Decoder network
def getDecoder(ip):

    u1 = UpSampling2D(size=(2, 2))(ip)
    d1 = conv2d(u1, ae_f*2)
    #
    u2 = UpSampling2D(size=(2, 2))(d1)
    d2 = conv2d(u2, ae_f)

    # Final
    d3 = Conv2D(1, (1, 1), activation='sigmoid')(d2)
    # print('Final:', d3._keras_shape)

    return d3

# get Model
def getModel(inputs, outputs, lrate=0.1, decay=1e-6, momentum=0.7):
    model = Model(inputs=inputs, outputs=outputs)

    sgd = SGD(lr=lrate, decay=decay, momentum=momentum, nesterov=True)
    model.compile(optimizer=sgd, loss=my_loss)

    return model

# Shared encoder network
def getNet():
    # Get encoder & decoders
    encInput, encOutput = getSharedEncoder(None, None, ipDepth)
    decOuput_A1 = getDecoder(encOutput)
    decOuput_A2 = getDecoder(encOutput)

    # Prepare model
    autoencoder_A1 = getModel(encInput, decOuput_A1, lrate, decay_A1)
    autoencoder_A2 = getModel(encInput, decOuput_A2, lrate, decay_A2)
    return autoencoder_A1, autoencoder_A2

################################ Data Processing and Utils #####################################
def crop(im, h, w):
    k = 0
    a = np.zeros((4,h,w,3), dtype=np.uint8)
    for i in range(0,im.shape[0]-h+1, h):
        for j in range(0,im.shape[1]-w+1, w):
            a[k,:,:,:] = im[i:i+h, j:j+w, :]
            k +=1
    return a

# Preprocess Data
def preprocessData(imArray):
    imgHeight, imgWidth, cDepth = imArray.shape

    # print('Image Array shape before : ', imArray.shape)

    # pad to fit network
    p = 8
    padX = (p - imgHeight % p)
    padY = (p - imgWidth % p)
    temp = np.zeros((imgHeight + padX, imgWidth + padY, cDepth), dtype=np.uint8)
    temp[:imgHeight, :imgWidth, :] = imArray

    imgHeight, imgWidth, cDepth = temp.shape

    # pad edge values
    temp1 = np.zeros((1, imgHeight + 80, imgWidth + 80, cDepth), dtype=np.uint8)
    for j in range(0,cDepth):
        temp1[0,:,:,j] = np.lib.pad(temp[:,:,j], (40), 'edge')

    imArray = temp1
    imArray = (imArray.astype('float32'))/ 255
    # print('Image Array shape after: ', imArray.shape)

    return imArray, padX, padY

################################################

# Weights path
weightsPath_A1 = "./weights/" + modelName + "/autoencoder_A1.hdf5"
weightsPath_A2 = "./weights/" + modelName + "/autoencoder_A2.hdf5"

# Get Network
autoencoder_A1, autoencoder_A2 = getNet()

# load weights
autoencoder_A1.load_weights(weightsPath_A1)
autoencoder_A2.load_weights(weightsPath_A2)

# Save path
savePath = datapath + modelName
if not os.path.exists(savePath):
    os.makedirs(savePath)

# Test: load images and predict
imgDataFiles = glob.glob(datapath+'*.*g')
imgDataFiles.sort()

for j in range(0, len(imgDataFiles)):

    print '-----------------------------> Image: ', j, ' <-----------------------------'
    # read image
    img = Image.open(imgDataFiles[j])
    img = np.asarray(img)
    img = np.expand_dims(img, axis=-1)

    # Process data: pad and normalize 
    imTest, PadX, PadY = preprocessData(img)
    imPred_A1 = autoencoder_A1.predict(imTest, batch_size=1)      # for noisy image
    imPred_A2 = autoencoder_A2.predict(imTest, batch_size=1)      # for clean image

    # Unpad
    imPred_A1 = np.squeeze(imPred_A1)
    imPred_A1 = imPred_A1[:-PadX, :-PadY]
    imPred_A2 = np.squeeze(imPred_A2)
    imPred_A2 = imPred_A2[:-PadX, :-PadY]

    # Save output
    # Image.fromarray(np.uint8((imPred_A1-np.min(imPred_A1))/(np.max(imPred_A1)-np.min(imPred_A1)) * 255)).save(savePath + '/A1_' +  os.path.basename(imgDataFiles[j]))
    Image.fromarray(np.uint8((imPred_A2-np.min(imPred_A2))/(np.max(imPred_A2)-np.min(imPred_A2)) * 255)).save(savePath + '/A2_' +  os.path.basename(imgDataFiles[j]))

print('--Done--')