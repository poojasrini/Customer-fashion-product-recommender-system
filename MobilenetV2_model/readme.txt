The scripts/notebooks and their purpose are as follows.

get_features.py - This script is used to generate feature vectors for images.

clustering.py - This script reads image feature vectors from a folder 
and saves the image similarity scores in json file. Builds ANNOY index based on cosine distance 
and crawls the index to get similar articles.

see_similar_results.ipynb - This notebook is to visualize similarity captured
between images.

test_predict.ipynb - This notebook is to test the recommendations made by this method.
Using mean average precision @k as the metric to quantify the model's 
performance.