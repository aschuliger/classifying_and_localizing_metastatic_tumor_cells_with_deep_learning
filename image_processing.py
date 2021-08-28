from PIL import Image
from libtiff import TIFF
import numpy as np
import glob
import sys
import cv2
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import math
from scipy.stats import mode

def preprocess_file(filename):
    print(filename)
    #processed_data = pd.read_csv(filename).groupby('plate').apply(aggr)
    try:
        img = Image.open(filename)
        #img = cv2.imread(filename)
        #img = TIFF.open(filename)
        #b,g,r = Image.Image.split(img)
        #imarray = [np.array(r),np.array(g),np.array(b)]
        imarray = np.array(img)
        #imarray = img.read_image()
        return imarray
    except:
        print("Bad file!")
        return 1

labels_file = pd.read_csv("_labels.csv")

path = "test_data"#"training_data/Final_project"#"test_images"
filenames = glob.glob(path + "/*.tif")
bad_files_count = 0
good_files_count = 0
for filename in filenames:
    id_name = filename[:-4][10:]
    #id_name = filename[:-4][12:]
    print(id_name)
    new_name = "preprocessed_test_data/prep_data_" + id_name + ".pickle"
    print(new_name)
    #print(filename[:-4][12:])
    data = preprocess_file(filename)

    if type(data) is int:
        bad_files_count = bad_files_count + 1
    else:
        good_files_count = good_files_count + 1
        #print(labels_file.loc[labels_file["id"]==id_name])
        label = labels_file.loc[labels_file["id"]==id_name]['label'].item()
        preprocessed_data = [data, label]
        print(data.shape)

        with open(new_name, 'wb') as f:
            pickle.dump(preprocessed_data, f)
print(bad_files_count)
print(good_files_count)