# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[training_hist]: ./report/training_hist.jpg "Training hist"
[validation_hist]: ./report/validation_hist.jpg "Validation hist"
[test_hist]: ./report/test_hist.jpg "Test hist"
[sample]: ./report/sampleimage.jpg "Sample"
[gray]: ./report/grayscale.jpg "Gray"
[histeq]: ./report/eq.jpg "Histeq"
[noise]: ./report/ns.jpg "Noise"
[final]: ./report/final.jpg "Final"

[30]: ./new_images/full_size/30kph_1.jpg "Traffic Sign 1"
[50]: ./new_images/full_size/50kph_2.jpg "Traffic Sign 2"
[yield]: ./new_images/full_size/yield_13.JPG "Traffic Sign 3"
[slip]: ./new_images/full_size/slippery_23.JPG "Traffic Sign 4"
[right]: ./new_images/full_size/turnright_33.JPG "Traffic Sign 5"


### Data Set Summary & Exploration

#### 1. Basic summary of the dataset

I used standard python & the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset

Let's have a look at the sign distributions in the training, validation and test sets:

![alt text][training_hist]

![alt text][validation_hist]

![alt text][test_hist]

We can see that some traffic signs are much more frequent than others (for instance the max speed 20 kph is 10 times less present than the very common 30 kph speed limit sign. I've never seen a 20kph max speed sign in 7 years in Germany ;-) ).
However the data distribution is very similar accross all data sets (training, validation and evaluation).
It would be interesting to see if the neural network performs better for the most frequently represented signs in the data set.

### Design and Test a Model Architecture

#### 1. Preprocessing

The preprocessing can be split into the following steps:
- Convert image to grayscale: Even though intuitively one may think that the color plays an important role in classifiying traffic signs, experimental results have shown that it was not necessarly the case. Therefore, to speed up the preprocessing process I decided to convert the images to grayscale.
- Perform histogram equalization: Some images from the training set have poor contrast, as a consequence I decided to add this step in the preprocessing routine.
- Add noise to the image: The network shall be applied to a number of images, coming from a diverse set of Cameras. In order to represent this diversity in the quality of the images, white noise was added to the image (+- 5). Such a step is also mentionned/recommended in the paper from Sermanet & LeCun.
- Normalize the image: Last step is normalizing the image, as required.

Let's have a look at each step in the plot below:

![alt text][sample] ![alt text][gray] ![alt text][histeq] ![alt text][noise]
![alt text][final]

#### 2. Model description

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale preprocessed image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Drop out           | 0.7 keep prob |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten            | Output 400 |
| Drop out           | 0.7 keep prob |
| Fully connected		| Output 120        									|
| RELU					|												|
| Drop out           | 0.7 keep prob |
| Fully connected		| Output 84        									|
| RELU					|												|
| Drop out           | 0.7 keep prob |
| Fully connected		| Output 43        									|

 
#### 3. Parameters used

Optimizer: Adam
Batch size: 128
Epochs: 50
Learning rate: 0.00025
Avg and Std for the initial weights: 0 and 0.1
Retain probability for drop out: 0.7

#### 4. How I found the solution

As one can easily recognize, my model is based on the LeNet5 architecture from the lab. It has proven to be already pretty good for the task of classifiying traffic signs. 

I tried several approaches in order to improve the performance of the LeNet5 for the required task. The first approach was to tune the preprocessing step in order to see what could be done there. What was in any case required was the normalization step. On top of that, I have compared the performance of the network on color and grayscale images. The difference was so minimal that I decided to continue the process only with grayscale images, since it speeds up the preprocessing process. After having seen a couple of sample images from the training set, I have decided to add a histogram equalization step in order to handle images with different contrasts. And finally adding noise to the training set prooved to be efficient in reducing over fitting.

To further reduce overfitting, I have decided to apply the dropout regularization technique. The idea behind it is to force the network to handle cases were some features are missing from the image (blurry edges, partially cropped sign etc...). Initially only at the input layer, later on each hidden layer too. Retaining probability was set to 0.7, and I have played around with the learning rate in order to optimize the performance of the network. Srivastavas' paper [insert_link] about dropout recommends in the Appendix that the learning rate be 10 to 100 times higher than the one from a standard network, which I have tried to apply. 

Pretty quickly I have reached a performance that was passing the minimal requirements, but I have unfortunately not had the time to try other architectures.

My final model results were:
* validation set accuracy of 94% (passes the minimal threshold of 93%)
* test set accuracy of 92%

### Test a Model on New Images

#### 1. Chosen imates

Here are five German traffic signs that I found on the web:

![alt text][30] ![alt text][50] ![alt text][yield] 
![alt text][slip] ![alt text][right]

All except the first image are actually cropped images from the google street view camera, which I thought would be representative enough in our case. 3 very common signs were chosen (30, 50 and yield) and two more rare images (sliperry road and right turn only). The slippery road and the right turn only signs are under represented in the training catalogue, as seen in the histograms above. I was curious to see if the network would still be able to correctly classify them. Before feeding them to the network the images were resized to fit the expected 32x32x1 shape.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h      		| 30 km/h     									| 
| 50 km/h     			| 50 km/h  										|
| Yield					| Yield											|
| Slippery Road			| Slippery Road      							|
| Right only	      		| Right only					 				|



The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of approx 92%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

Here are the top 5 softmax probabilities for each new sign:

* 30kph
| Label | Name | Probability
|:---------------------:|:---------------------------------------------:| 
| 1 | 30 km/h | .94 |
| 2 | 50 km/h | .05 |
| 0 | 20 km/h | 0 |
| 5 | 80 km/h | 0 |
| 6 | end of 80 km/h | 0 |

* 50kph
| Label | Name | Probability
|:---------------------:|:---------------------------------------------:| 
| 2 | 50 km/h | .71 |
| 5 | 80 km/h | .23 |
| 1 | 30 km/h | .03 |
| 3 | 60 km/h | .02 |
| 4 | 70km/h | 0 |

* yield
| Label | Name | Probability
|:---------------------:|:---------------------------------------------:| 
| 13 | Yield | 1 |
| 35 | Ahead only | 0 |
| 38 | Keep right | 0 |
| 3 | 60 km/h | 0 |
| 12 | Priority road | 0 |

* slippery road
| Label | Name | Probability
|:---------------------:|:---------------------------------------------:| 
| 23 | Sliperry road | .88 |
| 19 | Dangerous left curve | .10 |
| 31 | Wild animals crossing | .01 |
| 21 | Double curve | .01 |
| 30 | Beware of ice | 0 |

* right only
| Label | Name | Probability
|:---------------------:|:---------------------------------------------:| 
| 33 | Turn right ahead | 1 |
| 39 | Keep left | 0 |
| 37 | Go straight or right | 0 |
| 35 | Ahead only | 0 |
| 11 | Right of way | 0 |


One can notice that the top softmax probabilites for the speed limit signs are always related to other speed limit signs. Surprising is also the fact that the network is absolutely sure in it's classification for the yield and right only sign. The rest of the max probabilities are also pretty high (71% is the lowest, for 50 km/h). In conclusion the network performs excellently on the set of new images, and doesn't over fit the training data.
