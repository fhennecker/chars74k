import cv2
import argparse
import sys, os
import string
import re
import random

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

def get_class_index(filename):
    return int(re.findall(r'.*img(\d+).*', filename)[0])-1

def get_class(filename):
    return CLASSES[get_class_index(filename)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')

    opt = parser.parse_args()
    
    filenames = load_filenames(opt.datapath, ['Good', 'Bmp'])
    for f in random.sample(filenames, 10):
        print(f, get_class_index(f), get_class(f))
