This is a model for classifying the "EnglishImg" part of the
[Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).

* [Preprocessing](#preprocessing)
* [Model Choices](#model-choices)
* [Future leads](#leads-for-improving-performance)
* [Final model performance](#final-model-performance)
* [Failure analysis](#failure-analysis)
* [Help](#help)

## Preprocessing

#### Shape
All the images are of different shapes and sizes. Several methods were tried
to bring all the images to a common shape for training.

* Both 128x128 and 64x64 dimensions were tried; there wasn't much of a
  performance gain while using 128x128 so 64x64 was kept for faster training
* I tried keeping the aspect ratio of all images but that meant black bars
  had to be either side of the image, or above and below. Stretching the images
  actually helped towards getting better results overall.

#### Colour

* At first, the images were passed to the network in RGB but it was changed to
  grayscale as it didn't affect accuracy and grayscale might allow for 
  better generalisation. This also meant faster training.
* The RGB image is normalised channel-wise before being converted to grayscale.


#### Dataset split
The original dataset was split randomly the following way : 

* 60% training set (4623 samples)
* 20% validation set (1541 samples)
* 20% test set (1541 samples)

## Model choices
A convolutional neural network was chosen because they are known to perform
well for this type of problem.

* The baseline model was composed of a conv layer with 32 feature maps, 
  8x8 kernels and a 2x2 stride, followed by a fully connected layer and a
  softmax cross entropy loss. It obtained 43% accuracy on the validation set.
* Adding a second convolutional layer (with the same parameters as the 
  first one except for 4x4 kernels) pushed the accuracy to 60%. 
* A third convolutional layer also helped improving the accuracy but by a 
  small margin only.
* Increasing the number of feature maps in the second and third layer to 64
  and 128, and adding a hidden layer of 512 units just before the last
  fully connected layer pushed the performance to 65%.
* Dropout had been tried in several settings for the above architectures but
  didn't improve the results.
* Same had been done for pooling layers.

I did some reading up on rules of thumb when designing convnets and 
rewrote the network as follows:

    Conv1 : 3x3 kernels | 1x1 stride | 32 maps  | relu
    Conv2 : 5x5 kernels | 2x2 stride | 64 maps  | relu
    Conv3 : 5x5 kernels | 2x2 stride | 128 maps | relu
    Max Pooling : 2x2 kernels | 2x2 stride
    Fully Connected : 512 units | relu
    Dropout (0.8)
    Fully Connected : 62 units | softmax

This network gave the best results on the validation set at **71.7%**.

## Leads for improving performance
The performance obtained and discussed below are not state-of-the-art
([this guy](http://ankivil.com/kaggle-first-steps-with-julia-chars74k-first-place-using-convolutional-neural-networks/)
 claims for example to reach 86% accuracy). This is
because the scope of this challenge, I believe, was not to try to get
extremely close to it but rather to show a work flow and structure. However,
here are a couple of ideas to improve performance further :

* The dataset is quite small (less than 10K images). Data augmentation could
  help towards better generalisation (affine transformations on images, using
  the computer generated characters available in the original dataset, using
  GANs, ...)
* There are probably far better architectures, making better use of dropout,
  convolution and max pooling than the one presented here.

## Final model performance
The metric used is the percentage of correct predictions since this is a 
balanced multi-class prediction problem and we only care about whether we
predicted the right caracter or not. The chance level is at 1.61%.

The final test set accuracy is **69.3%**. This result is quite close to the
validation set accuracy (71.7%).

The confusion matrix shows quite good 
performance, but we can also see parallel lines to the diagonal; these lines
correspond to the lowercase/uppercase equivalent of the letter. 
(The y axis shows ground truth, and each row is normalised, and the x axis
shows predictions. The order of the classes, top to bottom, is : digits, 
uppercase, lowercase).

![Test set confusion matrix](https://raw.githubusercontent.com/fhennecker/chars74k/master/img/test_set_confusion.png)

An interesting metric is the case insensitive accuracy, which is **75.8%**. More
than 5% of the predictions are correct but miss the case.

A more informal metric is the top 3 accuracy : how many times is the correct
label in the top 3 predicted classes? This happened **86.5%** of the time in the
test set.

Prediction occured at more than 100FPS on a quad-core 2,3GHz Intel Core i7.

## Failure analysis
Here are randomly selected examples of errors in the test set classification.
Characters which were correctly predicted but had the wrong case prediction
were ignored on purpose. 

![](https://raw.githubusercontent.com/fhennecker/chars74k/master/img/misclassified1.png)
![](https://raw.githubusercontent.com/fhennecker/chars74k/master/img/misclassified2.png)

As you can see, some samples are of very bad quality (heavy distortion, zoom
or rotation), and some confusions can be understood (the cropped o being
interpreted as an 'n', the 'B' which has a similar structure as an 'E',...)

## Help
The requirements are python3, OpenCV and the requirements written in 
`requirements.txt` (run `pip install -r requirements.txt`)

    # run the dataset split (the English directory should be downloaded first)
    python preprocessing.py English -s datasplits/good

    # prepare directories for training
    mkdir saves summaries 

    # run training
    python train.py -t datasplits/good_train -v datasplits/good_validation -m saves/awesome_model

    # run testing
    python test.py -d datasplits/good_test -m saves/following_conventions-25000


