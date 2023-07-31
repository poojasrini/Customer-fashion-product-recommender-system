#====================================================================================
# This script is to convert all the images to feature vectors using VGG16(pretrained)
#====================================================================================

#====================================================================================
# Importing required libraries
#====================================================================================
from keras.utils import image_utils 
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import logging


#====================================================================================
# Necessary inputs
#====================================================================================
images_folder_path = "C:\\Personal\\h-and-m-personalized-fashion-recommendations\\h-and-m-personalized-fashion-recommendations\\images"


#====================================================================================
# Function for header of csv file
#====================================================================================
def write_header(writer,num_of_features):
    list=['image_filename']
    for i in range(1,num_of_features+1):
        list.append(f"feature_{str(i)}")
    writer.writerow(list)



# Defining the model
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

# Initializing the csv file to store image names and features
vgg16_file = open('vgg16.csv','w',newline='')
vgg16_writer = csv.writer(vgg16_file)

# Writing header for based on the the shape of the output of the model
num_of_features = model.layers[-2].output.shape[1]
write_header(vgg16_writer,num_of_features)

# List of subfolders in the images folder
sub_folders_list = os.listdir(images_folder_path)
num_sub_folders = len(sub_folders_list)

# Initializing log file
logging.basicConfig(filename='vgg16.log', filemode='w',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Processing image files
i = 0
for folder in sub_folders_list:
    sub_folders_path = os.path.join(images_folder_path,folder)
    file_list = os.listdir(sub_folders_path)
    for file in file_list:
        file_name = file.split('.')[0]
        file_path = os.path.join(sub_folders_path,file)
        image = image_utils.load_img(file_path, target_size=(224,224))
        image = image_utils.img_to_array(image) 
        reshaped_image = image.reshape(1,224,224,3) 
        imagex = preprocess_input(reshaped_image)
        image_features = model.predict(imagex, use_multiprocessing=True).tolist()[0]
        vgg16_writer.writerow([file_name]+image_features)
    i+=1
    logging.info(f'Processed {i} out of {num_sub_folders} subfolders') 

vgg16_file.close()

        






