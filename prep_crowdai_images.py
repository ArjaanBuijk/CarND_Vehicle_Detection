###################################################################################################
## Global settings & imports
###################################################################################################

I_AM_IN_JUPYTER = False
SCRATCH_IMAGE_DIR = 'C:\\Work\\ScratchImages'  # only used when exporting into .py, and setting I_AM_IN_JUPYTER=False
SCRATCH_IMAGE_NUM = 0

TRAIN_SVC = False  # set to false once svc is trained, and it will read cached pickle file

if I_AM_IN_JUPYTER:
    # get_ipython().magic('matplotlib inline')
    pass
else:
    # use non-interactive back-end to avoid images from popping up
    # See: http://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib-so-it-can-be
    from matplotlib import use
    use('Agg') 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from tqdm import tqdm
from sklearn.externals import joblib 
from scipy.ndimage.measurements import label
from collections import deque
import csv 
import pandas as pd 
import random

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

# NOTE: the next import is only valid for scikit-learn version <= 0.17
#from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split

###################################################################################################
## Utility functions for plotting images
###################################################################################################

# function to show a plot or write it to disk, depending if I am running in a jupyter notebook or not
def my_plt_show():
    global I_AM_IN_JUPYTER, SCRATCH_IMAGE_NUM, f_html, f_url
    plt.show()
    if I_AM_IN_JUPYTER == False:
        # at start
        if SCRATCH_IMAGE_NUM == 0:
            # clean out the scratch image dir
            files = glob.glob(SCRATCH_IMAGE_DIR+'\\*')
            for f in files:
                os.remove(f)  
            # open 'all.html' that displays all the images written
            f_html = open(SCRATCH_IMAGE_DIR+'\\all.html', 'w')
            f_url  = 'file:///'+SCRATCH_IMAGE_DIR+'\\all.html'
            f_html.write('<html>\n')
            # webbrowser.open_new(f_url) # open it in new window of default web-browser
            
        # save all images to a scratch dir
        fname = 'img_{:04d}.jpg'.format(SCRATCH_IMAGE_NUM)
        plt.savefig(SCRATCH_IMAGE_DIR+'\\'+fname)
        fig = plt.gcf() # get reference to the current figure
        plt.close(fig)  # and close it
        f_html.write('<img src="'+fname+'" /> <br />\n') 
        f_html.flush() # flush it directly to disk, for debug purposes.    
        # webbrowser.open(f_url, new=0) # refresh the page        
        SCRATCH_IMAGE_NUM += 1
    plt.gcf().clear() # clear the fig

# function to show an image with title
def show_image(image, title, cmap=None ):
    plt.gcf().clear() # clear the fig
    if I_AM_IN_JUPYTER:
        fig, ax = plt.subplots(1, 1, figsize=(24, 10))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.title(title)
    if cmap:
        plt.imshow(image, cmap=cmap) # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    else:
        plt.imshow(image)  
    my_plt_show()
    
    
# read the labeled file
# if it is a car, extract the window, scale to 64x64, and save as a png
file_dir_in  = './labeled_data/crowdai/object-detection-crowdai/'
file_dir_out = './labeled_data/vehicles/crowdai/'

# read sign labels into a panda DataFrame
df = pd.read_csv('./labeled_data/crowdai/labels_crowdai.csv')

create_cars = False
if create_cars:
    for i in range(len(df)):
        if df['Label'][i] == 'Car':
            file_in = file_dir_in+df['Frame'][i]
            xmin = df['xmin'][i]
            ymin = df['ymin'][i]
            xmax = df['xmax'][i]
            ymax = df['ymax'][i]
            
            if i < 3050 or xmin >= xmax or ymin >= ymax:
                continue
            
            image = mpimg.imread(file_in)
            
            #show_image(image, title=file_in)
            
            # Extract the car window from original image
            car_image = image[ymin:ymax, xmin:xmax]
            #show_image(car_image, title=file_in+'Car')
            
            # Resize it to 64x64
            car_image = cv2.resize(car_image, (64, 64))      
            #show_image(car_image, title=file_in+'Car - Resized')        
            
            # Save it as png
            file_out = file_dir_out+'image{:04d}.png'.format(i)
            mpimg.imsave(file_out, car_image)
        
create_not_cars = True
file_dir_no_cars  = './labeled_data/crowdai/images-to-create-non-vehicles/'
files_without_cars = os.listdir(file_dir_no_cars)

file_dir_out = './labeled_data/non-vehicles/Mine/'
if create_not_cars:
    for i in tqdm(range(len(df))):
        if df['Label'][i] == 'Car':
            file_in = file_dir_no_cars+files_without_cars[random.randint(0,len(files_without_cars)-1)]
            xmin = df['xmin'][i]
            ymin = df['ymin'][i]
            xmax = min(df['xmax'][i],1280)
            ymax = min(df['ymax'][i],720)
            
            if xmin >= xmax or ymin >= ymax:
                continue
            
            image = mpimg.imread(file_in)
            
            #show_image(image, title=file_in)
            
            # Extract the car window from original image
            car_image = image[ymin:ymax, xmin:xmax]
            #show_image(car_image, title=file_in+'Car')
            
            # Resize it to 64x64
            car_image = cv2.resize(car_image, (64, 64))      
            #show_image(car_image, title=file_in+'Car - Resized')        
            
            # Save it as png
            file_out = file_dir_out+'image{:04d}.png'.format(i)
            mpimg.imsave(file_out, car_image)
        
    

##with open('./labeled_data/crowdai/labels_crowdai.csv', 'r') as csvfile:
##    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
##    for row in spamreader:
##        print (', '.join(row))


