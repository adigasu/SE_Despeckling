"Shared Encoder based Denoising of OCT Images"

import os
import glob
import h5py as hp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
sEpoch = 1                  # Start epoch
eEpoch = 100                # End epoch
modelName = 'SE/'           # model name

backTrans = 1               # backtranslation flag
ipDepth = 1                 # channel
ipHeight = 220              # height
ipWidth = 220               # width
ae_f = 64                   # Autoencoder initial filter size

lrate = 0.1                 # Learning Rate
decay_A1 = 2e-5             # Weight decay of autoencoder 1 (A1)
decay_A2 = 6e-5             # Weight decay of autoencoder 2 (A2)
batch_sz = 8                # Batch size
delta = 0.85                # Loss function parameter
lflag = 1                   # Load weight flag

############################## Datapath ################################
datapath1 = './data/noisy_data/'
datapath2 = './data/clean_data/'
gtpath1 = datapath1
gtpath2 = datapath2

##############################################################
# Loss function
def loss_MS_SSIM(y_true, y_pred):

    # expected net output is of shape [batch_size, row, col, image_channels]
    # We need to shuffle this to [Batch_size, image_channels, row, col]
    y_true = y_true.dimshuffle([0, 3, 1, 2])
    y_pred = y_pred.dimshuffle([0, 3, 1, 2])

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = 1.0

    alpha = 1.0
    beta = 1.0
    gamma = 1.0

    for i in range(0,3):
        patches_true = K.T.nnet.neighbours.images2neibs(y_true, [10, 10], [5, 5])
        patches_pred = K.T.nnet.neighbours.images2neibs(y_pred, [10, 10], [5, 5])

        mx = K.mean(patches_true, axis=-1)
        my = K.mean(patches_pred, axis=-1)
        varx = K.var(patches_true, axis=-1)
        vary = K.var(patches_pred, axis=-1)
        covxy = K.mean(patches_true*patches_pred, axis=-1) - mx*my

        if i == 0:
            ssimLn = (2 * mx * my + c1)
            ssimLd = (mx ** 2 + my ** 2 + c1)
            ssimLn /= K.clip(ssimLd, K.epsilon(), np.inf)
            ssim = K.mean(ssimLn**(alpha * 3))

        ssimCn = (2 * K.sqrt(varx * vary  + K.epsilon()) + c2)
        ssimCd = (varx + vary + c2)
        ssimCn /= K.clip(ssimCd, K.epsilon(), np.inf)

        ssimSn = (covxy + c2/2)
        ssimSd = (K.sqrt(varx * vary + K.epsilon()) + c2/2)
        ssimSn /= K.clip(ssimSd, K.epsilon(), np.inf)

        ssim *= K.mean((ssimCn**beta) * (ssimSn**gamma))

        y_true = K.pool2d(y_true, (2,2),(2,2), data_format='channels_first', pool_mode='avg')
        y_pred = K.pool2d(y_pred, (2,2),(2,2), data_format='channels_first', pool_mode='avg')

    return (1.0 - ssim)

def my_loss(y_true, y_pred):
    ms_ssim_loss = loss_MS_SSIM(y_true, y_pred)
    l1_loss = K.mean(K.abs(y_pred - y_true))

    return delta * ms_ssim_loss + (1-delta) * l1_loss

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

    e1 = conv2d(inputs, ae_f)
    p1 = MaxPooling2D(pool_size=(2, 2))(e1)

    e2 = conv2d(p1, ae_f*2)
    p2 = MaxPooling2D(pool_size=(2, 2))(e2)
    
    e3 = conv2d(p2, ae_f*4)
    
    return inputs, e3

# Decoder network
def getDecoder(ip):

    u1 = UpSampling2D(size=(2, 2))(ip)
    d1 = conv2d(u1, ae_f*2)
    
    u2 = UpSampling2D(size=(2, 2))(d1)
    d2 = conv2d(u2, ae_f)

    # Final
    d3 = Conv2D(1, (1, 1), activation='sigmoid')(d2)

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
# Preprocess Data
def preprocessData(imArray):

    # [w, h, b] --> [b, h, w, ch], ch =1
    imgWidth, imgHeight, nbatch = imArray.shape
    imArray = np.transpose(imArray, (2,1,0))
    imArray = np.reshape(imArray, (nbatch, imgHeight, imgWidth, 1))
    imArray = (imArray.astype('float32'))/ 255

    return imArray

# Center crop image
def crop_center(img, cX, cY):
    x,y,n = img.shape
    sX = x//2-(cX//2)
    sY = y//2-(cY//2)
    return img[sX:sX+cX, sY:sY+cY, :]

# Process Data for back translation
def preprocessBTData(imArray):

    # [w, h, b] --> [b, h, w, ch], ch =1
    if len(imArray.shape) == 3:
        imgWidth, imgHeight, nbatch = imArray.shape
        imArray = np.transpose(imArray, (2,1,0))
        imArray = np.reshape(imArray, (nbatch, imgHeight, imgWidth, 1))

    nbatch, imgHeight, imgWidth, cDepth = imArray.shape

    # Pad array to fit network
    p = 8
    padX = (p - imgHeight % p)
    padY = (p - imgWidth % p)
    temp = np.zeros((nbatch, imgHeight + padX, imgWidth + padY, 1), dtype=np.uint8)
    temp[:, :imgHeight, :imgWidth, :] = imArray

    nbatch, imgHeight, imgWidth, cDepth = temp.shape

    # Pad edge values for maintain output size
    temp1 = np.zeros((nbatch, imgHeight + 80, imgWidth + 80, cDepth), dtype=np.uint8)
    for i in range(0, nbatch):
        for j in range(0, cDepth):
            temp1[i,:,:,j] = np.lib.pad(temp[i,:,:,j], (40), 'edge')

    # Normalize to [0,1]
    imArray = temp1
    imArray = (imArray.astype('float32'))/ 255
    # print('BT Image Array shape after: ', imArray.shape)

    return imArray, padX, padY

################################## Network Training functions #########################################
# Autoencoder train
def autoencoderTrain(X, Y, autoencoder, saveLossHist):

    autoencoder.fit(X, Y,
        epochs=1,
        batch_size=batch_sz,
        shuffle=True)

    lossHistory = autoencoder.history.history['loss']
    lossArray = np.array(lossHistory)

    with open(saveLossHist, 'a') as f:
        np.savetxt(f, lossArray, delimiter=",")

# Backtranslation train
def backTranslationTrain(bt_X, bt_Y, autoencoder1, autoencoder2):
    testX, PadX, PadY = preprocessBTData(bt_X)

    predX = autoencoder1.predict(testX, batch_size=1)
    predX = predX[:, :-PadX, :-PadY, :]

    autoencoder2.fit(predX, bt_Y,
        epochs=1,
        batch_size=batch_sz,
        shuffle=True)

# Autoencoder Test
def autoencoderTest(X, autoencoder):
    test, PadX, PadY = preprocessBTData(X)
    pred = autoencoder.predict(test, batch_size=1)
    pred = pred[:, :-PadX, :-PadY, :]
    return pred

################################# Load data and training ##################################################
# load images
imgDataFiles1 = glob.glob(datapath1 + '*.mat')
imgDataFiles1.sort()
imgGTFiles1 = glob.glob(gtpath1 + '*.mat')
imgGTFiles1.sort()
imgDataFiles2 = glob.glob(datapath2 + '*.mat')
imgDataFiles2.sort()
imgGTFiles2 = glob.glob(gtpath2 + '*.mat')
imgGTFiles2.sort()

# Get Network
autoencoder_A1, autoencoder_A2 = getNet()

print "---------------Final output of the network1: ----------------------"
autoencoder_A1.summary()
print "---------------Final output of the network2: ----------------------"
autoencoder_A2.summary()


# Save path (weights and loss values)
if not os.path.exists("./output_loss/"+ modelName):
  os.makedirs("./output_loss/"+ modelName)
if not os.path.exists("./weights/"+ modelName):
  os.makedirs("./weights/"+ modelName)

saveLossHist_A1 = "./output_loss/"+ modelName + str(sEpoch) + "_" + str(lrate) + "_autoencoder_A1.txt"
saveLossHist_A2 = "./output_loss/"+ modelName + str(sEpoch) + "_" + str(lrate) + "_autoencoder_A2.txt"


######################## Start training ###############################

for i in range(sEpoch, eEpoch):

    print '-----------------------------> Epochs: ', i, ' <-----------------------------'
    save_ae_A1 = "./weights/" + modelName + str(i) + "_autoencoder_A1.hdf5"
    save_ae_A2 = "./weights/" + modelName + str(i) + "_autoencoder_A2.hdf5"

    # Load weights at begining
    if (i == sEpoch) & (i != 1) & (lflag ==1):
        print('loading decoder weights...')
        print(' ')

        loadWeightsPath = "./weights/" + modelName + str(sEpoch-1) + "_autoencoder_A1.hdf5"
        autoencoder_A1.load_weights(loadWeightsPath)
        loadWeightsPath = "./weights/" + modelName + str(sEpoch-1) + "_autoencoder_A2.hdf5"
        autoencoder_A2.load_weights(loadWeightsPath)

        # Disable load flag
        lflag = 0

    # loop for each mat file
    for j in range(0, len(imgDataFiles1)):

        # Read data
        mat = hp.File(imgDataFiles1[j], 'r')
        Img_A1 = mat['data'].value

        mat = hp.File(imgGTFiles1[j], 'r')
        gt_A1 = mat['data'].value

        mat = hp.File(imgDataFiles2[j], 'r')
        Img_A2 = mat['data'].value

        mat = hp.File(imgGTFiles2[j], 'r')
        gt_A2 = mat['data'].value

        # Preprocess data: Normalize to [0,1]
        Img_A1_X = preprocessData(Img_A1)
        gt_A1 = crop_center(gt_A1, 140, 140)
        Img_A1_Y = preprocessData(gt_A1)

        Img_A2_X = preprocessData(Img_A2)
        gt_A2 = crop_center(gt_A2, 140, 140)
        Img_A2_Y = preprocessData(gt_A2)

        # Backtranslation training (skip for first epoch)
        if (backTrans == 1) & (i != 1):
            backTranslationTrain(Img_A2, Img_A2_Y, autoencoder_A1, autoencoder_A2)
            backTranslationTrain(Img_A1, Img_A1_Y, autoencoder_A2, autoencoder_A1)

        # Train autoencoders
        autoencoderTrain(Img_A1_X, Img_A1_Y, autoencoder_A1, saveLossHist_A1)
        autoencoderTrain(Img_A2_X, Img_A2_Y, autoencoder_A2, saveLossHist_A2)

        # Save weights every 2 epochs
        if (i%2 == 0):
            autoencoder_A1.save_weights(save_ae_A1)
            autoencoder_A2.save_weights(save_ae_A2)

######################## End training ###############################
