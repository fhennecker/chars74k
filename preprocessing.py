import cv2
import argparse
import os
import string, re
import numpy as np
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

def get_class_index(filename):
    return int(re.findall(r'.*img(\d+).*', filename)[0])-1

def get_class(filename):
    """Get the actual digit or character of the image"""
    return CLASSES[get_class_index(filename)]

def get_batch(dataset, batch_size, dimensions):
    batch_filenames = random.sample(dataset, batch_size)
    images = np.array(list(map(lambda f:open_image(f, dimensions), batch_filenames)))
    labels = np.array(list(map(get_class_index, batch_filenames)))
    return images, labels

def images_stats(dataset):
    """Print max width and height of all images in dataset"""
    max_width, max_height = 0, 0
    for img_name in dataset:
        img = cv2.imread(img_name)
        if img.shape[0] > max_height: max_height = img.shape[0]
        if img.shape[1] > max_width: max_width = img.shape[1]
    print("Max width : %d, max height: %d" % (max_width, max_height))

def open_image(filename, scale_to=[64, 64]):
    """Opens an image, returns the preprocessed image (scaled, masked)"""
    img = cv2.imread(filename) * cv2.imread(filename.replace('Bmp', 'Msk'))/255
    processed_img = np.zeros(list(scale_to)+[3])

    # scaling
    #  img_w, img_h = img.shape[1], img.shape[0]
    #  target_w, target_h = scale_to[1], scale_to[0]
    #  factor = target_w / img_w if img_w/img_h > target_w/target_h else target_h / img_h
    #  img = cv2.resize(img, None, fx=factor, fy=factor)
    img = cv2.resize(img, tuple(scale_to))

    # centering image
    #  x, y = int(target_w/2 - img.shape[1]/2), int(target_h/2 - img.shape[0]/2)
    #  processed_img[y:y+img.shape[0], x:x+img.shape[1]] = img

    # normalising
    processed_img = img.astype(np.float32)
    for c in range(3):
        processed_img[:,:,c] /= np.max(processed_img[:,:,c])

    # to grayscale
    processed_img = cv2.cvtColor(
            (processed_img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    processed_img = np.expand_dims(processed_img, -1)

    return processed_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')
    parser.add_argument('-s', default="", help='Split and save dataset')
    parser.add_argument('-t', action='store_true', help='Print dataset stats')

    opt = parser.parse_args()
    
    if opt.s: 
        filenames = load_filenames(opt.datapath, ['Good', 'Bmp'])
        split_and_save_dataset(filenames, opt.s)
    if opt.t: images_stats(filenames)


