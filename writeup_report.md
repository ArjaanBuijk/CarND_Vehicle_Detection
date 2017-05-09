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

    ![track1](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/videoProject.gif?raw=true)
  
---

# 2. Utility functions

In code cell 2, there are some utility functions to plot images, and a function to draw lines onto an image.

---

# 3. Histogram of Oriented Gradents (HOG) and color features

The code to extract features from the images can be found in code cells 3.

For the extraction of HOG features, I use the hog function of skimage.feature. This function is nicely described in the scikit-image documentation ([<b>hog</b>](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html)). 

There are several parameters to drive the extraction of hog features, and I simply ran a series of experiments to get the best result for my classifier. This exercise is described below.

The functions in code cell 3 take inputs to try out all variations of color space, orientations, pixels_per_cell and cells_per_block.

In addition to HOG features, I also make use of binned color features and histograms of the color channels.

---

# 4. Training of the classifier & selection of color space

Code to train a linear SVM classifier can be found in cell 4.

I augmented the data set with the ([<b>Udacity data</b>](https://github.com/udacity/self-driving-car/tree/master/annotations)). It turned out augmenting with this data improved detection of the white car coming into view. 

In order to use the Udacity data, I wrote a small ([<b>python script</b>](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/prep_crowdai_images.py)) that reads the csv file, extracted the window region of the car, scaled the size to 64x64, and saved it as a *.png file. This provided an addition 72k images. To balance it out, I selected images without cars from the project video and the harder challenge project video of project 4, and extracted the same amount of non-vehicle pictures.

When training the SVM with all the images, my computer ran into a memory limit. I reduced the number of training samples to 35,000 for vehicles and 35,000 for non-vehicles.

After extracting all the features for cars and notcars, I normalize them using the StandardScalar module of the sklearn package.

I had initially selected the HLS color space, because it gave the highest accuracy during training & testing of the SVM on the training set. However, during use on the project video, it had a tendency to think that the yellow lines in the picture are cars. This is when I switched to the YCrCb color space, and a lot of the false positives went away.

I kept 20% of the total training data as test data, and the accuracy of the trained SVM was 98.8 % when applied to this test data. 

Reading the images and training the SVM takes a bit of time, about 15-20 minutes total. So, I cached the trained model, and the scaler information to disk, using pickle.

In the submitted notebook, the flag TRAIN_SVC is set to False, so it will not train the SVM, but read the data from the saved pickle files. 


# 5. Sliding Window Search & Thresholding logic

The sliding window search and the thresholding logic is found in code cell 5, 6, 7 and 8. 

To check what it is doing, it is best to look at the full sequence of images produced by the very last code cell in the Jupyter notebook ([HTML version](https://github.com/ArjaanBuijk/CarND_Vehicle_Detection/blob/master/P5.html))

The  process_image takes an rgb image, and calls the function find_cars, which was mostly taken as is from the course example, with addition of a heatmap thresholding logic to eliminate false positives.

The nice thing about the find_cars function is that it extracts the HOG features for the entire image only once, and then loops over patches equal to the size of a sliding window. This makes the whole process much faster. My first implementation used an actual sliding window search, extracting the HOG features each time for the window, and the find_cars implementation was about 6x faster!

I tried different scales and different cells_per_step (which controls the overlap of patches), and settled on a scale of 1.5 and a cell_per_step of 2. That was giving me the best result and I could filter out almost every remaining false positive with a heatmap thresholding.

The thresholding I created uses a class to store the hot-windows of the current and 3 previous images. I am storing the hot-windows of 4 images, and each time a new image is stored, the oldest one is pushed out automatically. I am using a deque to accomplish this.

In code cell 7, the thresholding is done. I calculate the heatmap for the current and 3 previous images. (Note, this can be optimized, but I leave this for later...). Then, I create a combined binary threshold from these heatmaps, by requiring that a pixel must have a value larger than a threshold in each subsequent heatmap. I put the threshold to 1. This  means that if a pixel is part of two overlapping car-windows, for 4 images in a row, it is accepted.

After heatmap thresholding, the boxes to draw are found using the label function of the scipy.ndimage.measurements package.

 

# 6. Summary

The end result can be summarized as follows:

- The classifier is able to detect the black car very good. 
- The classifier detects the white car less robustly, but still quite good.
- The classifier does detect false positives, but the heatmap thresholding over several frames filters out most of them.
- Every now and then the filtering does not fully work, and for a couple frames some false positives result in a flicker of a falsely detected car window.


The main opportunities I see to make it even more robust against detection of false positives are:

- Further improve the classifier. Probably by using more or better training data for non-vehicles.
- Further improve the heatmap thresholding. 

And lastly, there is room for improvement to detect a larger bounding box for the full vehicle, while avoiding false positives.