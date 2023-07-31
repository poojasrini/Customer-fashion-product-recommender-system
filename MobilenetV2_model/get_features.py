import tensorflow as tf
import tensorflow_hub as hub
import keras
import pandas as pd
import pathlib
import PIL
import zipfile
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import time


images_ds = "/home/as6608/aml/proj/images.zip"
features_path = "/home/as6608/aml/proj/image_feat/"

def preprocess(img):

  # img = tf.convert_to_tensor(img)
  img = tf.keras.utils.img_to_array(img)

  # # Decodes the image to W x H x 3 shape tensor with type of uint8
  # img = tf.io.decode_jpeg(img, channels=3)

  # Resize the image to 224 x 224 x 3 shape tensor
  img = tf.image.resize_with_pad(img, 224, 224)

  # Converts the data type of uint8 to float32 by adding a new axis
  # This makes the img 1 x 224 x 224 x 3 tensor with the data type of float32
  # This is required for the mobilenet model we are using
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

  return img

#################################################
# This function:
# Loads the mobilenet model from TF.HUB
# Makes an inference for all images stored in a local folder
# Saves each of the feature vectors in a file
#################################################
def get_image_feature_vectors():
  start_time = time.time()
  print("---------------------------------")
  print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Started at %s" %time.ctime())
  print("---------------------------------")

  # Definition of module with using tfhub.dev handle
  module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/5" 
  
  # Load the module
  module = hub.load(module_handle)

  print("---------------------------------")
  print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Completed at %s" %time.ctime())
  print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))

  print("---------------------------------")
  print ("Step.2 of 2 - Generating Feature Vectors -  Started at %s" %time.ctime())
 

  # Loops through all images in a local folder
  imgzip = zipfile.ZipFile(images_ds)
  inflist = imgzip.infolist()[21000:]

  i = 0
  for f in inflist:
    
    if (f.filename != '061/0616100001.jpg'):
      print("file:", f.filename)
      i += 1
      # Loads and pre-process the image
      ifile = imgzip.open(f)
      img = Image.open(ifile)
      img = preprocess(img)

      # Calculate the image feature vector of the img
      features = module(img)   
    
      # Remove single-dimensional entries from the 'features' array
      feature_set = np.squeeze(features)  

      # Saves the image feature vectors into a file for later use
      outfile_name = f.filename.split('/')[1].split(".")[0] + ".npz"
      out_path = features_path + outfile_name

      # Saves the 'feature_set' to a text file
      np.savetxt(out_path, feature_set, delimiter=',')

      print("Image feature vector saved to   :%s" %out_path)
    
  print("---------------------------------")
  print ("Step.2 of 2 - Generating Feature Vectors - Completed at %s" %time.ctime())
  print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
  print("--- %s images processed ---------" %i)
    
get_image_feature_vectors()