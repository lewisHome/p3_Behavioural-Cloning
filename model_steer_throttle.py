import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

   
samples = []
with open('driving_log_trottle.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Lambda
from keras.layers import Conv2D, Dropout
from keras.layers import Cropping2D, Concatenate
from keras.layers.pooling import MaxPooling2D

# Augmentation functions taken from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def shadow_aug(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def translate_image(image,tr_x,tr_y):
    # Translation
    rows = image.shape[0]
    cols = image.shape[1]
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_translate = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_translate

def translate_angle(angle, tr_x, trans_range):
    angle_translate = angle + tr_x/trans_range*2*.2   
    return angle_translate

def translate_throttel(throttle, tr_x, tr_y, trans_range):
    throttel_translate = throttle - (tr_x/trans_range)*.2 - (tr_y/trans_range)*.4
    if throttel_translate > 1:
        throttel_translate = 1
    elif throttel_translate < 0:
        throttel_translate = 0
    return throttel_translate

def generator(samples, batch_size):
    num_samples = len(samples)
    angle_correction = 0.2
    throttel_correction = 0.2
    while 1: # Loop forever so the generator never terminates
        samples=sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            throttels = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = '.git/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
                    #crop image to show road only
                    if i == 0:
                        angle = float(batch_sample[3])
                        throttel = float(batch_sample[4])
                    elif i == 1:
                        angle = float(batch_sample[3]) + angle_correction
                        throttel = float(batch_sample[4])-throttel_correction
                        if throttel <0:
                            throttle = 0
                    elif i == 2:
                        angle = float(batch_sample[3]) - angle_correction
                        throttel = float(batch_sample[4])-throttel_correction
                        if throttel <0:
                            throttel = 0

                    #origional data
                    images.append(image)
                    angles.append(angle)
                    throttels.append(throttel)
                    #flipped data
                    images.append(cv2.flip(image,1))
                    angles.append(angle*-1.0)
                    throttels.append(throttel)
                    #translate values
                    trans_range=60
                    tr_x = trans_range*np.random.uniform()-trans_range/2
                    tr_y = 40*np.random.uniform()-40/2
                    images.append(translate_image(image, tr_x, tr_y))
                    angles.append(translate_angle(angle, tr_x, trans_range))
                    throttels.append(translate_throttel(throttel, tr_x, tr_y, trans_range))
                    #translate flipped data
                    images.append(translate_image(cv2.flip(image,1), tr_x, tr_y))
                    angles.append(translate_angle(angle*-1.0, tr_x, trans_range))
                    throttels.append(translate_throttel(throttel, tr_x, tr_y, trans_range))
                    #shadow augmentation
                    images.append(shadow_aug(image))
                    angles.append(angle)
                    throttels.append(throttel)
                    #shadow flipped data
                    images.append(shadow_aug(cv2.flip(image,1)))
                    angles.append(angle*-1.0)
                    throttels.append(throttel)
                    

            X_train = np.array(images)
            y_train1 = np.array(angles)
            y_train2 = np.array(throttels)
            yield (X_train, [y_train1, y_train2])

# compile and train the model using the generator function
genBatchSize = 64
train_generator = generator(train_samples, batch_size=genBatchSize)
validation_generator = generator(validation_samples, batch_size=genBatchSize)


inputImg = Input(shape =(160,320,3))
crop = Cropping2D(cropping=((50,20), (0,0)))(inputImg)
normalisation =Lambda(lambda x: (x/255.0)-0.5)(crop)
conv1 = Conv2D( 24, (5, 5), strides=(2,2), activation="relu")(normalisation)
conv2 = Conv2D( 36, (5, 5), strides=(2,2), activation="relu")(conv1)
conv3 = Conv2D(48,(5,5),strides=(2,2), activation = "relu")(conv2)
conv4 = Conv2D(64,(3,3),activation="relu")(conv3)
conv5 = Conv2D(64,(3,3),activation="relu")(conv4)
Flat = Flatten()(conv5)

FC1s = Dense(100)(Flat)
DP1s = Dropout(0.25)(FC1s)
FC2s = Dense(50)(DP1s)
DP2s = Dropout(0.25)(FC2s)
FC3s = Dense(10)(DP2s)
DP3s = Dropout(0.25)(FC3s)
steering = Dense(1)(DP3s)

FC1a = Dense(100)(Flat)
DP1a = Dropout(0.25)(FC1a)
FC2a = Dense(50)(DP1a)
DP2a = Dropout(0.25)(FC2a)
FC3a = Dense(10)(DP2a)
DP3a = Dropout(0.25)(FC3a)
acceleration = Dense(1)(DP3a)

trainStep = int(len(train_samples*18)/genBatchSize)
print("Training steps per epoch:", trainStep)
valStep = int(len(validation_samples)/genBatchSize)
print("Validation steps per epoch:", valStep)

model = Model(inputs = inputImg, outputs = [steering, acceleration])
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator,
                    steps_per_epoch = trainStep,
                    validation_data = validation_generator,
                    validation_steps = valStep,
                    epochs = 3,
                    verbose = 1)

model.save('model_steer_throttle.h5')
print("Model Saved!")
