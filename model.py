import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

   
samples = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Dropout
from keras.layers import Cropping2D
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

def generator(samples, batch_size=128):
    num_samples = len(samples)
    correction = 0.2   
    while 1: # Loop forever so the generator never terminates
        samples=sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = batch_sample[i].split('/')[-1]
                    image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
                    #crop image to show road only
                    if i == 0:
                        angle = float(batch_sample[3])
                    elif i == 1:
                        angle = float(batch_sample[3])+correction
                    elif i == 2:
                        angle = float(batch_sample[3])- correction
                    #origional data
                    images.append(image)
                    angles.append(angle)
                    #flipped data
                    images.append(cv2.flip(image,1))
                    angles.append(angle*-1.0)
                    #translate values
                    trans_range=60
                    tr_x = trans_range*np.random.uniform()-trans_range/2
                    tr_y = 40*np.random.uniform()-40/2
                    images.append(translate_image(image, tr_x, tr_y))
                    angles.append(translate_angle(angle, tr_x, trans_range))
                    #translate flipped data
                    images.append(translate_image(cv2.flip(image,1), tr_x, tr_y))
                    angles.append(translate_angle(angle*-1.0, tr_x, trans_range))
                    #shadow augmentation
                    images.append(shadow_aug(image))
                    angles.append(angle)
                    #shadow flipped data
                    images.append(shadow_aug(cv2.flip(image,1)))
                    angles.append(angle*-1.0)
                    

            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape =(160,320,3)))
model.add(Lambda(lambda x: (x/255.0)-0.5))
model.add(Convolution2D( 24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D( 36, 5, 5, subsample=(2,2), activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5,subsample=(2,2), activation = "relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Dropout(0.25))
model.add(Dense(1))


model.compile(loss='mse', optimizer='Nadam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples*18), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=2)

model.save('model.h5')
print("Model Saved!")
