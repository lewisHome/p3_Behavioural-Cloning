# Behavioral Cloning Project

The code and writeup contained in this repository are submitted for the third Udacity self driving car nano-degree project. The aim of the project is to collect driver behaviour data and use it to train a neural network which will then be able to drive a car around a track. 

For a successful submission in this project it must be shown that the trained network is able to drive the car around the track once. A second track is also avaliable in the simulator and if possible it would be good if the car could drive around this track too.

The approach I took to the second track was that becuase it is not necessary to complete a lap of the second track I would try and train my network on the first track only and see if I could get it to drive around the second track based off of that experience.

## Prior Work

NVIDIA developed an [end to end method of driving a car](https://arxiv.org/abs/1604.07316) using a Convolutional Neural Network to control the steering of a car. That is they trained a network that took images from a central forward facing camera as an input and used steering angle as the trainable output. They posit that this method will eventually allow for better performance from smaller systems as the network internally learns to look for important road features. This approach is in contrast to the method of a human predefining features to look for such as lane markings etc.

## Data Collection

To train an end to end network to drive a car we need data and this is collected using the [Udacity driving simulator](https://github.com/udacity/self-driving-car-sim).
![SimulatorEnviroment](examples/SimImage.JPG)
The car is driven around the track using the mouse to control the steering and your arrow keys to control the throttle and brakes. The simulator records the image data and the telemetry data. The telemetry data is recorded in a csv file formatted as shown in the table below.

Centre Image | Left Image | Right Image | Steering Angle | Throttle Value | Brake Value | Speed Value 
-------------|-------------|-------------|-------------|-------------|-------------|-------------
IMG/center_2016_12_01_13_32_43_457.jpg | IMG/left_2016_12_01_13_32_43_457.jpg | IMG/right_2016_12_01_13_32_43_457.jpg | 0.0617599 | 0.9855326 | 0 | 2.124567

Initially I tried to collect my own data. I recorded 1 lap of the car driving in each direction. The track is essentially a loop and by collecting data with the car driving in both directions this ensured that my data did not have a bias towards left or right hand turns. As I progressed through the project I found that the quality of the data I collected was not very good, that is I was not very good at driving the car in the simulator. To overcome this I trained my final model using the [Udacity data set](https://www.dropbox.com/s/2mfk5a2v2zymr3e/Dataset.zip?dl=0).

## Data Augmentation

For me an important lesson from the previous project Traffic sign classifier was that for data augmentation to be worthwhile it must be done in a manner such that the augmented data generated is realistic and relevant. For example randomly flipping images upside down will be of no benefit. The first method of data augmentation I used utilised the cameras on the sides of the vehicle which are recorded concurrently with the central camera image from the simulator. This allows me to collect data in which it appears the vehicle is vearing the side of the track. Further to this I leaned heavily on an article by [Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) on the data augmentation techniques he used when he did this project. I shamelessly borrowed two techniques the first was to simulate greater angles and gradients and the second was to simulate shadows over the track.

### Side View Cameras

Left Camera Image | Centre Camera Image | Right Camera Image
------------------|------------------|------------------|
![Left Image](examples/left_2016_12_01_13_32_43_457.jpg)|![Centre Image](examples/center_2016_12_01_13_32_43_457.jpg) |![Right Image](examples/right_2016_12_01_13_32_43_457.jpg)

To generate the steering angles for the side view cameras I applied a correction factor to the steering angles.

    correction = 0.2 
    for i in range(3):
        name = batch_sample[i].split('/')[-1]
        image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
        if i == 0:
            angle = float(batch_sample[3])
        elif i == 1:
            angle = float(batch_sample[3]) + correction
        elif i == 2:
            angle = float(batch_sample[3]) - correction
            

### Angle and Gradient Simulation



## Network Architecture


Here is a video of my car driving around the test track using my first the car as it drives around the track.
![Track1Video](examples/Track1.mp4)









I decided as a challenge I would only record data from the first track and see if I could develop a network that was robust enough to drive around the second track with letting the network see it first. As you can see visually the track is very different, it also has sharper corners and larger changes in track gradient.
![Second Track](examples/SimImage2.JPG)
To help with my challenge I referenced this excellent article by . My key take away from Vivek's article is that when augmenting data it must be realistic. To that end I borrowed two of his augmentation techniques. The first was to generate augmented images with random horizontal and vertical shifts. By introducing horizontal shifts I am trying to train the network to cope with sharper corners. Likewise by introducing vertical shifts I am trying to train the network to deal with changes in gradient on the road. The primary test track is a very gentle track to drive around with a maximum steering angle of roughly 5&deg; 

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
