# Behavioral Cloning Project

The code and writeup contained in this repository are submitted for the third Udacity self driving car nano-degree project. The aim of the project is to collect driver behaviour data and use it to train a neural network which will then be able to drive a car around a track. 

For a successful submission in this project it must be shown that the trained network is able to drive the car around the track once. A second track is also avaliable in the simulator and if possible it would be good if the car could drive around this track too.

The approach I took to the second track was that becuase it is not necessary to complete a lap of the second track to submit the project I thought I would have some fun with it. Therefore I decided I would only train my network with data collected from the first track and see if I could derive a model which was generalisable enough to drive around the second track aswell. To try to accomplish this task I first tried to augment the data to simualte the conditions of the second track. That included simulating sharper corners, steeper road angles and shadows across the track. I found that data augmentation alone was not enough so I also built a second network to predict both the throttle and the steering angle.

Test Track for Submission | Second Test Track
--------------------------|--------------------------
![SimulatorEnviroment](examples/SimImage.JPG)|![Second Track](examples/SimImage2.JPG)


## Prior Work

NVIDIA developed an [end to end method of driving a car](https://arxiv.org/abs/1604.07316) using a Convolutional Neural Network to control the steering of a car. That is they trained a network that took images from a central forward facing camera as an input and used steering angle as the trainable output. They posit that this method will eventually allow for better performance from smaller systems as the network internally learns to look for important road features. This approach is in contrast to the method of a human predefining features to look for such as lane markings etc.

## Data Collection

To train an end to end network to drive a car we need data and this is collected using the [Udacity driving simulator](https://github.com/udacity/self-driving-car-sim).
The car is driven around the track using the mouse to control the steering and your arrow keys to control the throttle and brakes. The simulator records the image data and the telemetry data. The telemetry data is recorded in a csv file formatted as shown in the table below.

Centre Image | Left Image | Right Image | Steering Angle | Throttle Value | Brake Value | Speed Value 
-------------|-------------|-------------|-------------|-------------|-------------|-------------
IMG/center_2016_12_01_13_32_43_457.jpg | IMG/left_2016_12_01_13_32_43_457.jpg | IMG/right_2016_12_01_13_32_43_457.jpg | 0.0617599 | 0.9855326 | 0 | 2.124567

Initially I tried to collect my own data. I recorded 1 lap of the car driving in each direction. The track is essentially a loop and by collecting data with the car driving in both directions this ensured that my data did not have a bias towards left or right hand turns. As I progressed through the project I found that the quality of the data I collected was not very good, that is I was not very good at driving the car in the simulator. To overcome this I trained my final model using the [Udacity data set](https://www.dropbox.com/s/2mfk5a2v2zymr3e/Dataset.zip?dl=0).

## Data Augmentation

For me an important lesson from the previous project Traffic sign classifier was that for data augmentation to be worthwhile it must be done in a manner such that the augmented data generated is realistic and relevant. For example randomly flipping images upside down will be of no benefit. However flipping the images through the vertical axis will be of benefit as it simulates driving around corners in the opposite diretion. 

The first method of data augmentation I used utilised the cameras on the sides of the vehicle which are recorded concurrently with the central camera image from the simulator. This allows me to collect data in which it appears the vehicle is vearing the side of the track. I also flipped all images through the vertical axis and inverted the steering angles. Further to this I leaned heavily on an article by [Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) on the data augmentation techniques he used when he did this project. I shamelessly borrowed two techniques the first was to simulate greater angles and gradients and the second was to simulate shadows over the track.

### Side View Cameras

Centre Camera Image | Left Camera Image | Right Camera Image
------------------|------------------|------------------|
![Centre Image](examples/center_2016_12_01_13_32_43_457.jpg) |![Left Image](examples/left_2016_12_01_13_32_43_457.jpg)|![Right Image](examples/right_2016_12_01_13_32_43_457.jpg)

To generate the steering angles for the side view cameras I applied a correction factor to the steering angles.

    #steering correction factor
    correction = 0.2 
    #loop through three image as listed in driving log
    for i in range(3):
        #load filename from driving log
        name = batch_sample[i].split('/')[-1]
        
        #import image using opencv and convert colour space from BGR to RGB
        image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
        
        #for the centre image the steering angle 
        #is that which has been recorded
        if i == 0:
            angle = float(batch_sample[3])
        
        #for the left camera image the car needs 
        #to steer to the right so the correction factor
        #is added to the recorded steering angle
        elif i == 1:
            angle = float(batch_sample[3]) + correction
            
        #for the right camera image the car needs
        #to steer to the left so the correction factor
        #is subtracted from the recorded steering angle            
        elif i == 2:
            angle = float(batch_sample[3]) - correction
            
This additional data allows the car to learn what to do if it the vehicle is not aligned with the road centre line.

### Flipping Images Through Vertical Axis

As mentioned earlier flipping images through the veritcal axis allows us to simulate left hand corners as right hand corners and vise versa.

I utilised the opencv function fliplr to accomplish the image manipulation

    imageFlip = cv2.flip(image,1)
   
I inverted the steering angle simply by multiplying the steering angle by the float -1.0

    angle = angle*-1.0

Image Input | Flipped Image
------------|------------
![Input Image](examples/center_2016_12_01_13_33_08_039.jpg)|![Flipped Image](examples/center_2016_12_01_13_33_08_039_flip.jpg)|
Steering Angle| Flipped Steering Angle
0.243562 | -0.243562

### Angle and Gradient Simulation

To further suplement the data I applied translations to the image to simulate the car driving on gradients by applying vertical translations to the image data. I also applied horizontal translations to the image data to further supplement data for cornering.

To translate the image I defined a function to take an image and shift it a given number of pixels horizontally and vertically using the opencv function warpAffine.

    def translate_image(image,tr_x,tr_y):
        # Translation
        rows = image.shape[0]
        cols = image.shape[1]
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        image_translate = cv2.warpAffine(image,Trans_M,(cols,rows))
        return image_translate

A second function is defined to generate a steering angle to accompany the translated image. The constants applied to the translation function are taken directly from Vivek's blog post.

    def translate_angle(angle, tr_x, trans_range):
        angle_translate = angle + tr_x/trans_range*2*.2   
        return angle_translate

    trans_range=60
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = 40*np.random.uniform()-40/2
    images.append(translate_image(image, tr_x, tr_y))
    angles.append(translate_angle(angle, tr_x, trans_range))
                    
Input Image | Translated Image Example 1 | Translated Image Example 2 | Translated Image Example 3
------------|------------|------------|------------
![Centre Image](examples/center_2016_12_01_13_32_43_457.jpg) |![Image Translate 0](examples/center_2016_12_01_13_32_43_457_translate_0.jpg)|![Image Translate 1](examples/center_2016_12_01_13_32_43_457_translate_1.jpg)|![Image Translate 2](examples/center_2016_12_01_13_32_43_457_translate_2.jpg)
Steering Angle | Augmented Steering Angle | Augmented Steering Angle | Augmented Steering Angle 
0.0617599 | 0.2400561374026167 | -0.030398056564315934 | 0.04933732241641371


### Random Shadow Augmentation

I also copied Vivek Yadavs method of simulating shadows across the image. There are changes in the colour of the road however this augmentation technique is more relevant to the second track where there are a number of points around the track where shadows are cast accross the road.

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


Input Image | Translated Image Example 1 | Translated Image Example 2 | Translated Image Example 3
------------|------------|------------|------------
![Centre Image](examples/center_2016_12_01_13_32_43_457.jpg) |![Image Shadow 0](examples/center_2016_12_01_13_32_43_457_shadow_0.jpg)|![Image Shadow 1](examples/center_2016_12_01_13_32_43_457_shadow_1.jpg)|![Image Shadow 2](examples/center_2016_12_01_13_32_43_457_shadow_2.jpg)

This augmentation technique does not simulate a shift in the camera and therefore the steering angle which accompanies this augmentation technique is that recorded for the centre camera or derived for the left and right hand cameras as outlined earlier.
 
## Network Architecture



## Results
### NVIDIA Network on First Test Track
Here is a video of my car driving around the test track using my first the car as it drives around the track.
![Track1Video](examples/Track1.mp4)

### NVIDIA Network on Second Test Track

### Steering and Speed Network on First Test Track

### Steering and Speed Network on Second Test Track


## Conclusions




