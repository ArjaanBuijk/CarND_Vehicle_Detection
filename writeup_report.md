# Vehicle Detection and Tracking

---

Vehicle Detection and Tracking Project 

---

The goals / steps of this project are the following:

* Train a classifier, using HOG and color features, to tell if an image represents a car or not
* Apply the trained classifier to detect cars in images of a video stream, and draw a box around them

All the steps are explained & documented in this [Jupyter notebook ](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/P5.ipynb). (or it's [HTML version](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/P5.html))


The write up you are reading provides some additional information.

---

# 1. Submission includes all required files

My project includes the following files:

- [<b>P5.ipynb</b> - The Jupyter notebook containing all code with explanations](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/P5.ipynb)
- [<b>P5.html</b> - The executed Jupyter notebook as HTML ](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/P5.html)
- [<b>writeup_report.md</b> - A summary of the project](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/writeup_report.md)
- [<b>videoProject.mp4</b> - The project video with boxes drawn around cars](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/videoProject.mp4)

<b>Animated gif of the video:</b>
    ![track1](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/videoProject.gif?raw=true)
  
---

# 2. Utility functions

In code cell 2, there are some utility functions to plot images, and a function to draw lines onto an image.

---

# 3. Histogram of Oriented Gradients (HOG) and color features

The code to extract features from the images can be found in code cell 3.

For the extraction of HOG features, I use the hog function of skimage.feature. This function is nicely described in the scikit-image documentation ([<b>hog</b>](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html)). 

There are several parameters to drive the extraction of hog features. To get the best performance of the classifier, I simply ran a series of experiments.

The functions in code cell 3 take inputs to try out all variations of color space, orientations, pixels_per_cell and cells_per_block.

The HOG features can be visualized, as shown here for a car and a non-car images from the training set, where the HOG features are shown for each color channel of the YCrCb color space. The car image is from the [<b>Udacity data set</b>](https://github.com/udacity/self-driving-car/tree/master/annotations), which I used in addition to the provided training images.

<b>Example of HOG Features for a Car Image</b>

Original Image: ![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/car-image0000.png?raw=true)

![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/hog-image-car-1.JPG?raw=true)

<b>Example of HOG Features for a Not Car Image</b>

Original Image: ![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/not-car-image0000.png?raw=true)

![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/hog-image-not-car-1.JPG?raw=true)

In addition to HOG features, I also make use of binned color features and histograms of the color channels. All these features are combined into a single feature vector that represents a training image.

By doing this for every training image, we get a large set of feature vectors. To avoid that a certain feature (HOG, binned color or histogram) is dominating all the others, the features are normalized. This is done with the StandardScalar function of the sklearn package.


---

# 4. Training of the classifier & selection of color space

Code to train a classifier can be found in cell 4.

I augmented the data set with the ([<b>Udacity data</b>](https://github.com/udacity/self-driving-car/tree/master/annotations)). It turned out augmenting with this data improved detection of the white car coming into view. 

In order to use the Udacity data, I wrote a small ([<b>python script</b>](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/prep_crowdai_images.py)) that reads the csv file, extracts the window region of the car, scales the size to 64x64, and saves it as a png file. This provided an addition 72,000 images. To balance it out with the same amount of non-car images, I took some images without cars from the project video and extracted the same amount of non-vehicle training examples.

When training the classifier with all the images, my computer ran into a memory limit. I reduced the number of training samples to 35,000 vehicles and 35,000 non-vehicles.

I tested two classifiers:

1. Linear SVM
2. MLP

The choice what classifier to use can be set in code cell 1, with the USE_SVC or USE_MLP flag.

I kept 20% of the total training data as test data, and used the rest as training data. 

Reading the images and training the classifier takes a bit of time, about 15-20 minutes total. So, I cached the trained model, and the scaler information to disk, using pickle.

In the submitted notebook, the flag TRAIN_CLASSIFIER is set to True, as is USE_SVC, so it will train the Linear SVM classifier.

Both classifiers gave me similar results. I ended up selecting the SVM because it performed better on detecting the white car in some of the later images. I believe though that either classifier is a good choice and can be made to work for this project.

<b> Selecting parameters for the SVM Classifier</b> 

I had initially selected the HLS color space, because it gave the highest accuracy during training & testing of the SVM on the training set. However, during use on the project video, it had a tendency to think that the yellow lines in the picture are cars. This is when I switched to the YCrCb color space, and many of the false positives went away.

The parameters I used are:

| parameter | value |
| --------- | ----- |
| color_space     | 'YCrCb'|   
| hog_channel     | 'ALL'  |   
| orient          | 9       |       
| pix_per_cell    | 8       | 
| cell_per_block  | 2      |
| spatial_size    | (32, 32)| 
| hist_bins       | 32      |    



<b> Accuracy of the trained SVM Classifier</b>

The accuracy of the trained SVM was <b>98.69 %</b>.
 


# 5. Sliding Window Search & Thresholding logic

The sliding window search and the thresholding logic is found in code cell 5, 6, 7 and 8. 

The function process_image takes an rgb image as input, and calls the function find_cars, which was mostly taken as is from the course example, with addition of a heatmap thresholding logic to eliminate false positives.

The nice thing about the find_cars function is that it extracts the HOG features for a large region of the entire image only once, and then loops over that region with overlapping windows which size can be defined by a scale factor. This makes the whole process much faster. My first implementation used the method where the HOG features were extracted each time for the sliding window, and the find_cars implementation was about 6x faster!

I tried different search regions with scales of search windows and also tested the influence of the cells_per_step parameter, which controls the overlap of the search windows.

<b> Selected regions, scales and overlaps for HOG feature extraction</b>

| What | Values |
| ---- | -----  |
| cells_per_step | 1 |
| region 1: ystart, ystop, scale | 395, 550, 1.25 |
| region 2: ystart, ystop, scale | 500, 656, 1.50 |

<b>Decision Thresholding</b>

To deal with the false positives, I tried using a threshold on the decision function of the SVM itself, with the clf.decision_function capability. This function returns a confidence score based on how far away the sample is from the decision boundary. This way, low confidence predictions can be filtered out, which in theory should filter away false positives and leave the true positives in place. 

It turned out though that in some images, the SVM was very confident in predicting false positives, and trying to filter them out using this decision thresholding approach was not going to work. Setting a tight tolerance would work for that particular image, but it resulted in filtering away true positives in other images.

I abandoned this approach and relied purely on the heatmap thresholding described in the next section.


<b>Heatmap Thresholding</b>


In code cell 7, the heatmap thresholding is implemented, using the following parameters:


| Parameter | Value | Description |
| --------- | ----- | ----------- |
| N_HOT_WINDOWS | 17 | Number of frames for which hot windows are stored for heatmap thresholding |
| HEAT_THRESHOLD_1 | 4 | Heatmap threshold for individual frames. Pixels that are inside this number of overlapping hot windows are set to 1, all others are set to 0. We end up with a binary thresholded image. |
| HEAT_THRESHOLD_2 | 14 | Number of frames that a pixel must be above the individual threshold HEAT_THRESHOLD_1. This does not have to be in sequential frames, but it has to be met within the N_HOT_WINDOWS saved frames. This allows a car detection to fail a maximum of (N_HOT_WINDOWS - HEAT_THRESHOLD_2) times, and we still do not lose track of the car. This threshold step results in a combined, binary thresholded image |

In other words, with these settings, we accept any pixel that was in 4 overlapping hot windows, 14 times in the last 17 frames as belonging to a car.

It also means, that it takes at least 14 frames before a car will be detected, which is quite long, but acceptable. If we would improve the classifier, and reduce the prediction of false positives, we can shorten this number of required frames to accept a car detection.

We then pass the combined, binary thresholded image into the <b>label</b> function of the scipy.ndimage.measurements package, to determine the bounding boxes that must be drawn on the image.

And the final step is to draw the bounding boxes and return the image. 


To visualize how the thresholding is working, please look at the full sequence of images produced by the very last code cell in the Jupyter notebook ([HTML version](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/P5.html)). In that code cell, I run the logic on frames 741-760, and the following can be seen in the key frames 741 and 754:

<b>Frame 741: start of visualization demo</b> 


- SVM detects hot-windows for both cars and a few false positives
- Binary Heatmap Threshold for this image eliminates some of the false positives, but not all
- History of heatmaps not yet build up, so combined binary heatmap threshold is still completely empty
- No Bounding Boxes drawn yet on image
 
![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/frame-741-01-with-hot-windows.png?raw=true)

![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/frame-741-02-heatmap.png?raw=true)

![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/frame-741-03-thresholded.png?raw=true)

![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/frame-741-04-combined-threshold.png?raw=true)

![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/frame-741-05-with-bounding-boxes.png?raw=true)

<b>Frame 754: first accepted detection of cars</b>

With the threshold settings chosen, it takes 14 frames before the combined binary threshold accepts a selection as a true positive for a car, and the first bounding box is drawn.

- SVM detects hot-windows for both cars and a few false positives
- Binary Heatmap Threshold for this image eliminates some of the false positives, but not all
- History of heatmaps now fully build up, and combined binary heatmap threshold eliminates all false positives, while leaving the true positives for the cars in place
- The Bounding Boxes are drawn on the image


![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/frame-754-01-with-hot-windows.png?raw=true)

![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/frame-754-02-heatmap.png?raw=true)

![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/frame-754-03-thresholded.png?raw=true)

![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/frame-754-04-combined-threshold.png?raw=true)

![Image](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/images/frame-754-05-with-bounding-boxes.png?raw=true)

 

# 6. Summary

The end result can be summarized as follows:

- The classifier is able to detect both the white & black car very good in individual frames.
- The classifier does detect a fair amount of false positives, sometimes with high confidence.
- An elaborate heatmap thresholding over 14-17 frames was needed to eliminate the false positives while keeping the true positives.


Even though it works well for this video, there is definitely a lot of room for improvement. I feel there are way too many false positives and rely too heavily on an elaborate heatmap thresholding which is brittle. I do not expect the current logic to work well on other cases without having to do significant work in fine-tuning the classifier and the thresholding.

Concretely, to improve it, I would focus on:

- Optimize the classifier. 
- Use more or better training data, especially for non-vehicles.
