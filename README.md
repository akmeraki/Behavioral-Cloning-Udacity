# Behavioral-Cloning
## Self-Driving Car Nanodegree

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<p align="center">
<img src="https://github.com/akmeraki/Behavioral-Cloning-Udacity/blob/master/Images/ezgif.com-gif-maker.gif">
</p>



### Overview
The objective of this project is to clone human driving behavior by use of Convolutional Neural Network in a simulated driving application, we are going to use a Udacity developed Car Simulator using Unity game engine. The simulator consists of training and Autonomous modes to rive the car. First, during the training session, we will navigate our car inside the simulator using the keyboard or a joystick for continuous motion of the steering angle. While we navigating the car the simulator records training images and respective steering angles. Then we use those recorded data to train our neural network. It uses the trained model to predict steering angles for a car in the simulator given a frame from the central camera.

### Dependencies

This project requires **Python 3.6** and the following Python libraries installed:
Please utilize the environment file to install related packages.

- [Environment File](https://github.com/akmeraki/Behavioral-Cloning-Udacity/tree/master/Environment)
- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [Scikit-learn](http://scikit-learn.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [OpenCV2](http://opencv.org/)

### Files in this repo
- model.py - The script used to build and train the model.
- drive.py - The script to drive the car.
- model.json - The model architecture.
- model.h5 - The model weights.
- model-test1.json model-test1.h5 -

### Simulator

You can download the stimulator from this Udacity repository : [Udacity Self Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)

## Implementation

### How to Generate Model

You can set the folder path at line 17 in model.py make sure that your folder contains \IMG\ (generated images) folder and a csv (driving data)
- `data_dirs = ['folder_name']`

### My driving data

[Training Data](https://drive.google.com/file/d/0Bw2un6-T5az-R3JxQ1lFUDlQRzQ/view?usp=sharing) <br>
[Udacity test Dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) <br>

### How to Run the Model

You can clone the repo and run the drive.py file with the stimulator in Autonomous mode.
- `python drive.py model.json`

## Network Architecture
The network consists of three convolutional layers. The layers are increasing in depth decreasing in size. There are three fully connected layers Dropout is employed between the fully connected layers and activation function is relu. <br>
Here is the network architecture as shown by keras

<table style="width: 42px;">
<tbody>
<tr style="height: 42px;">
<th style="width: 10px; height: 42px;">Layer (type)&nbsp;&nbsp;</th>
<th style="width: 10px; height: 42px;">&nbsp;Output Shape</th>
<th style="width: 10.2px; height: 42px;">Param #&nbsp;</th>
<th style="width: 10px; height: 42px;">&nbsp;Connected to</th>
</tr>
<tr style="height: 62px;">
<td style="width: 10px; height: 62px;">&nbsp;convolution2d_1 (Convolution2D)</td>
<td style="width: 10px; height: 62px;">&nbsp;(None, 40, 80, 32)</td>
<td style="width: 10.2px; height: 62px;">&nbsp;320</td>
<td style="width: 10px; height: 62px;">convolution2d_input_1[0][0]</td>
</tr>
<tr style="height: 42px;">
<td style="width: 10px; height: 42px;">&nbsp;maxpooling2d_1 (MaxPooling2D)</td>
<td style="width: 10px; height: 42px;">&nbsp;(None, 20, 40, 32)</td>
<td style="width: 10.2px; height: 42px;">&nbsp;0</td>
<td style="width: 10px; height: 42px;">convolution2d_1[0][0]</td>
</tr>
<tr style="height: 42px;">
<td style="width: 10px; height: 42px;">&nbsp;convolution2d_2 (Convolution2D)&nbsp;</td>
<td style="width: 10px; height: 42px;">&nbsp;(None, 10, 20, 64)</td>
<td style="width: 10.2px; height: 42px;">18496&nbsp;</td>
<td style="width: 10px; height: 42px;">maxpooling2d_1[0][0]</td>
</tr>
<tr style="height: 42px;">
<td style="width: 10px; height: 42px;">&nbsp;maxpooling2d_2 (MaxPooling2D)&nbsp;</td>
<td style="width: 10px; height: 42px;">&nbsp;(None, 10, 20, 64)</td>
<td style="width: 10.2px; height: 42px;">0&nbsp;</td>
<td style="width: 10px; height: 42px;">convolution2d_2[0][0]</td>
</tr>
<tr style="height: 22.4px;">
<td style="width: 10px; height: 22.4px;">&nbsp;convolution2d_3 (Convolution2D)&nbsp;</td>
<td style="width: 10px; height: 22.4px;">&nbsp;(None, 10, 20, 128)</td>
<td style="width: 10.2px; height: 22.4px;">73856&nbsp;</td>
<td style="width: 10px; height: 22.4px;">maxpooling2d_2[0][0]</td>
</tr>
<tr style="height: 22px;">
<td style="width: 10px; height: 22px;">&nbsp;maxpooling2d_3 (MaxPooling2D)</td>
<td style="width: 10px; height: 22px;">&nbsp;(None, 5, 10, 128)</td>
<td style="width: 10.2px; height: 22px;">0&nbsp;</td>
<td style="width: 10px; height: 22px;">convolution2d_3[0][0]</td>
</tr>
<tr style="height: 22px;">
<td style="width: 10px; height: 22px;">&nbsp;flatten_1 (Flatten)</td>
<td style="width: 10px; height: 22px;">(None, 6400)&nbsp;</td>
<td style="width: 10.2px; height: 22px;">0</td>
<td style="width: 10px; height: 22px;">maxpooling2d_3[0][0]</td>
</tr>
<tr style="height: 22px;">
<td style="width: 10px; height: 22px;">&nbsp;dense_1 (Dense) &nbsp;</td>
<td style="width: 10px; height: 22px;">&nbsp;(None, 500)</td>
<td style="width: 10.2px; height: 22px;">&nbsp;3200500</td>
<td style="width: 10px; height: 22px;">flatten_1[0][0]</td>
</tr>
<tr style="height: 22px;">
<td style="width: 10px; height: 22px;">&nbsp;dropout_1 (Dropout)</td>
<td style="width: 10px; height: 22px;">&nbsp;(None, 500)</td>
<td style="width: 10.2px; height: 22px;">&nbsp;0</td>
<td style="width: 10px; height: 22px;">dense_1[0][0]</td>
</tr>
<tr style="height: 22px;">
<td style="width: 10px; height: 22px;">&nbsp;dense_2 (Dense)&nbsp;</td>
<td style="width: 10px; height: 22px;">&nbsp;(None, 100)</td>
<td style="width: 10.2px; height: 22px;">50100&nbsp;</td>
<td style="width: 10px; height: 22px;">dropout_1[0][0]</td>
</tr>
<tr style="height: 22px;">
<td style="width: 10px; height: 22px;">&nbsp;dropout_2 (Dropout) &nbsp;&nbsp;</td>
<td style="width: 10px; height: 22px;">&nbsp;(None, 100)</td>
<td style="width: 10.2px; height: 22px;">0&nbsp;</td>
<td style="width: 10px; height: 22px;">dense_2[0][0]</td>
</tr>
<tr style="height: 22px;">
<td style="width: 10px; height: 22px;">dense_3 (Dense)&nbsp;</td>
<td style="width: 10px; height: 22px;">(None, 10)</td>
<td style="width: 10.2px; height: 22px;">1010</td>
<td style="width: 10px; height: 22px;">dropout_2[0][0]</td>
</tr>
<tr style="height: 22px;">
<td style="width: 10px; height: 22px;">dropout_3 (Dropout)&nbsp;</td>
<td style="width: 10px; height: 22px;">(None, 10)</td>
<td style="width: 10.2px; height: 22px;">0</td>
<td style="width: 10px; height: 22px;">dense_3[0][0]</td>
</tr>
<tr style="height: 22px;">
<td style="width: 10px; height: 22px;">dense_4 (Dense)</td>
<td style="width: 10px; height: 22px;">(None, 1)</td>
<td style="width: 10.2px; height: 22px;">11</td>
<td style="width: 10px; height: 22px;">dropout_3[0][0]</td>
</tr>
</tbody>
</table>

## List of Contents
- Collecting Data
- Balancing Data
- Data Preprocessing
- Defining the Architecture
- Connecting Simulator to model
- Data Augmentation Techniques
- Batch Generators
- Tuning Experiments
- Results

## Collecting Data

Obtaining good training data, is crucial for the model to run well. As, we will see in the results section that, this was the main problem I faced while training my model. How well the neural network drives depends on how well we drive the car in the training mode.
The training data can be collected from the udacity simulator in the training mode.
I tried to keep the car in the middle of the road for the entire laps.
Four laps in the forward direction and four laps in the reverse direction would be enough for the training data. It is crucial to do it in both ways, as we don't want the model to have a steering bias as the forward direction mostly contains only left turns.

[Dowload the Udacity training data here.
](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

# Balancing the Data

The training data obtained can be visualized in the form of a histogram
to understand how the training data is structured. The histogram is divided by 25 bins, and each bin represents a steering angle ranging from -1 to 1 .

<p align="center">
<img src="https://github.com/akmeraki/Behavioral-Cloning-Udacity/blob/master/Images/Training_data.png">
</p>

As seen from histogram the raw data shows that most of the angles are near the zero angle as the car is driven in the middle of the road at an angle close to zero most of the time.
Hence to reduce the zero angle bias significantly, but still preserve a little zero angle bias since we want the car to drive in the middle of the road.

<p align="center">
<img src="https://github.com/akmeraki/Behavioral-Cloning-Udacity/blob/master/Images/Histogram_of_training.png">
</p>

## Data Preprocessing

There are three main steps to data preprocessing:
- Data is split into Training and validation data and visualized.
- Color space conversion - the image color space is converted from RGB to YUV format,as this helps the nvidia model to learn better as prosed from the nividia paper.
- Resizing the image from (320px, 160px) original size to (200px, 66px) using OpenCV resize.
- Normalization - scaling the data to the range of 0-1

<p align="center">
<img src="https://github.com/akmeraki/Behavioral-Cloning-Udacity/blob/master/cropped_images/Comparison%20of%20Preprocessed%20and%20Original%20Image.png">
</p>


I choose only one channel (Satuation) as it might reduce the burden of proceessing and in turn reduce the computational power reduired.

## Data Augmentation techniques

We utilize many Augmentation techniques as the training data generated in the generator is not enough to train the model.
We use many image Augmentation Techniques like zooming, panning, changing brightness, flipping .

#### Comparison of Zoomed and original Image
<p align="center">
<img src="https://github.com/akmeraki/Behavioral-Cloning-Udacity/blob/master/cropped_images/Comparison%20of%20Zoomed%20and%20Original%20Image.png">
</p>

#### Comparison of Panned and original Image

<p align="center">
<img src="https://github.com/akmeraki/Behavioral-Cloning-Udacity/blob/master/cropped_images/Comparison%20of%20Panning%20and%20Original%20Image.png">
</p>


#### Comparison of brightness altered and original Image

<p align="center">
<img src="https://github.com/akmeraki/Behavioral-Cloning-Udacity/blob/master/cropped_images/Comparison%20of%20random_brightness_image%20and%20Original.png">
</p>

#### Comparison of Flipped and original Image

<p align="center">
<img src="<p align="center">
<img src="https://github.com/akmeraki/Behavioral-Cloning-Udacity/blob/master/cropped_images/Comparison%20of%20Flipped%20Image%20and%20Original%20Image.png">
</p>


## Batch Generators

There are two multithreaded nested generators supporting train, val and test sets. The outer batch generator called threaded_generator which consists of producer and consumer
launches the inner batch generator called batch_generator in a separate thread and
caches 10 output.


#### Displaying the Augmented batch generated Training data

<p align="center">
<img src="<p align="center">
<img src="https://github.com/akmeraki/Behavioral-Cloning-Udacity/blob/master/Images/comparison%20of%20Augmented%20images%20with%20the%20Original%20Images%20.png">
</p>

To support three data types the batch generator accepts the batch size
a second parameter that selects the type such as train val or test.
Based on this parameter one of three csv file arrays are chosen. The arrays are
prepared earlier in the data loading phase where all the csv files are read.
It is also made to support multiple directory so different datasets could could be used.
and the rows are merged in one array. Then the array is split into parts train
test and val and assigned to different variable. This approach simplifies data
shuffling since the csv rows contain both features and labels and are small in
size.

Batch Generator now randomly samples batch_size rows from the array, reads the
images from respective files, preprocesses the images and appends them to
images array (X). The labels are appended to labels array and if three
cameras are used the labels for left and right cameras are adjusted by 0.1
and -0.1 respectively.

## Experiments

- Based on the Cheatsheet (https://carnd-forums.udacity.com/cq/viewquestion.action?id=26214464&questionTitle=behavioral-cloning-cheatsheet) i tired keeping the epoch low as 5. But it didnt work in my case unless its 18-20 so i kept it to the max 20.

- Tried to reduce the dropout layers as the there was a significantly large gap between training and validation loss curves.

- With better data added into the Training data, the car able to generalize better.

- Removing relu activations on the architecture and adding elu (exponential linear units) instead since this takes care of the vanishing gradient problem.

- I tried to make my model work in track 2 it was unsuccessful. Its due to my models choice.

- I tried to work as simalar to the NVIDIA Model but that had 5 convolution layers, but to optimize mine i finilized to 3 convolution layers each.

## Results and Conclusion

Like almost all machine learning projects, this is project is no exception, Our model is only as good as our data is. The key to this project is to have a good training data. There were instances in this project that were quite frustrating to be honest, because my car would sway to edge of the lane, and jump off the road. I added in a few recovery laps, this solved the problem. Now the network had learned to sway back to the middle of the road, if it had gone too close to the edge.

This project, I was working on a regression problem in the terms of self-driving cars. We mainly focused on finding a suitable network architecture and trained a model a dataset. According to Mean Square Error (MSE) the model worked well. Next generating new new dataset was the problem here. Additionally, we didn't fully rely on MSE when building our final model. Also, we use relatively number of training epochs (namely 20 epochs).

![Click here to view the complete Video of the Self-Driving car in Youtube.](https://www.youtube.com/watch?v=mjknidprRE)

[![Final Output](https://github.com/akmeraki/Behavioral-Cloning-Udacity/blob/master/Images/Self_Driving_Car_Nanodegree_Program.png)](https://www.youtube.com/watch?v=mjknidprREo)
