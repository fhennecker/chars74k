import cv2
import argparse
import sys, os
import string
import re
import random
import numpy as np

CLASSES = string.digits + string.ascii_uppercase + string.ascii_lowercase

def load_filenames(datapath, filters=[]):
    """Loads all files contained in datapath where path contains all strings
    contained in filters"""

    filenames = []
    for path, dirs, files in os.walk(datapath):
        # filtering out unwanted paths
        if sum(map(lambda f : f in path, filters)) == len(filters):
            filenames += list(map(lambda f : path+'/'+f, files))
    return filenames

def split_and_save_dataset(dataset, filename):
    """Splits a dataset in 3 and saves lists of filenames."""
    splits = [0.6, 0.2, 0.2]
    split_names = ['train', 'validation', 'test']
    perm = np.random.permutation(len(dataset))
    
    for s, split in enumerate(splits):
        startindex = int(sum(splits[:s]) * len(dataset))
        endindex = int(startindex + splits[s] * len(dataset))
        with open(filename+'_'+split_names[s], 'w') as f:
            for i in perm[startindex:endindex]:
                f.write(dataset[i]+'\n')

    with open(filename, 'w') as f:
        for name in dataset:
            f.write(name+'\n')

def get_class_index(filename):
    return int(re.findall(r'.*img(\d+).*', filename)[0])-1

def get_class(filename):
    """Get the actual digit or character of the image"""
    return CLASSES[get_class_index(filename)]

def images_stats(dataset):
    """Print max width and height of all images in dataset"""
    max_width, max_height = 0, 0
    for img_name in dataset:
        img = cv2.imread(img_name)
        if img.shape[0] > max_height: max_height = img.shape[0]
        if img.shape[1] > max_width: max_width = img.shape[1]
    print("Max width : %d, max height: %d" % (max_width, max_height))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')
    parser.add_argument('-s', action='store_true', help='Split and save dataset')

    opt = parser.parse_args()
    
    filenames = load_filenames(opt.datapath, ['Good', 'Bmp'])
    if opt.s:
        split_and_save_dataset(filenames, 'good')
    images_stats(filenames)


