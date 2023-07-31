#################################################
# This script reads image feature vectors from a folder
# and saves the image similarity scores in json file
#################################################

import numpy as np
import time
import json
from annoy import AnnoyIndex
from scipy import spatial
import glob
import os
from joblib import Parallel, delayed
#################################################

#################################################
# This function; 
# Reads all image feature vectores stored in /feature-vectors/*.npz
# Adds them all in Annoy Index
# Builds ANNOY index
# Calculates the nearest neighbors and image similarity metrics
# Stores image similarity scores in a json file
#################################################

def cluster():

  start_time = time.time()
  features_path = "/home/as6608/aml/proj/image_feat/"
  # features_path = "/content/drive/MyDrive/AML/test/"
  similarity_file = "/home/as6608/aml/proj/nearest_neighbors2.json"

  
  print("---------------------------------")
  print ("Step.1 - ANNOY index generation - Started at %s" %time.ctime())
  print("---------------------------------")

  # Defining data structures as empty dict
  file_index_to_file_name = {}
  file_index_to_file_vector = {}

  # Configuring annoy parameters
  dims = 1792
  n_nearest_neighbors = 10
  trees = 10000

  # Reads all file names which stores feature vectors 
  allfiles = os.listdir(features_path)
  print("Got all files")

  t = AnnoyIndex(dims, metric='angular')
  t.verbose(True)
  t.on_disk_build('sim_idx.ann')

  for file_index, i in enumerate(allfiles):
    # Reads feature vectors and assigns them into the file_vector 
    file_vector = np.loadtxt(features_path+i)

    file_name = os.path.basename(i).split('.')[0]
    file_index_to_file_name[file_index] = file_name
    file_index_to_file_vector[file_index] = file_vector

    # Adds image feature vectors into annoy index   
    t.add_item(file_index, file_vector)

    print("---------------------------------")
    print("Annoy index     : %s" %file_index)
    print("Image file name : %s" %file_name)
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))


  # Builds annoy index
  t.build(trees, n_jobs=8)

  print ("Step.1 - ANNOY index generation - Finished")
  print ("Step.2 - Similarity score calculation - Started ") 
  
  named_nearest_neighbors = []

    # Loops through all indexed items
  for i in file_index_to_file_name.keys():
    master_file_name = file_index_to_file_name[i]
    print("for file:", master_file_name)
    master_vector = file_index_to_file_vector[i]

    # Calculates the nearest neighbors of the master item
    nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)

    # Loops through the nearest neighbors of the master item
    for j in nearest_neighbors:

      # Assigns file_name, image feature vectors and product id values of the similar item
      neighbor_file_name = file_index_to_file_name[j]
      neighbor_file_vector = file_index_to_file_vector[j]

      # Calculates the similarity score of the similar item
      similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
      rounded_similarity = int((similarity * 10000)) / 10000.0

      # Appends master product id with the similarity score 
      # and the product id of the similar items
      if master_file_name != neighbor_file_name:
        named_nearest_neighbors.append({
          'similarity': rounded_similarity,
          'master_pi': master_file_name,
          'similar_pi': neighbor_file_name})

    print("---------------------------------") 
    print("Similarity index       : %s" %i)
    print("Master Image file name : %s" %file_index_to_file_name[i]) 
    print("Nearest Neighbors.     : %s" %nearest_neighbors) 
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
  # Parallel(n_jobs=8, require="sharedmem")(delayed(getnns)(i) for i in file_index_to_file_name.keys())
  
  print ("Step.2 - Similarity score calculation - Finished ") 

  with open(similarity_file, 'w') as out:
    json.dump(named_nearest_neighbors, out)

  print ("Step.3 - Data stored in 'nearest_neighbors.json' file ") 
  print("--- Process completed in %.2f minutes ---------" % ((time.time() - start_time)/60))

cluster()