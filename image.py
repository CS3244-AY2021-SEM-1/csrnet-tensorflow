import random
import os
from PIL import Image, ImageFilter, ImageDraw, ImageStat
import numpy as np
import h5py
import cv2

def create_img(path):
    """
    path: image path
    Function to load,normalize and return image 
    """
    img = Image.open(path).convert('RGB')
    
    img = np.array(im)
    
    img = im/255.0
    
    img[:,:,0]=(img[:,:,0]-0.485)/0.229
    img[:,:,1]=(img[:,:,1]-0.456)/0.224
    img[:,:,2]=(img[:,:,2]-0.406)/0.225

    return img

def get_input(path):
    path = path[0] 
    img = create_img(path)
    return(img)

def get_output(path):
    """
    path: ground truth path
    import ground truth density map and resize it
    """
    
    gt_file = h5py.File(path,'r')
    
    target = np.asarray(gt_file['density'])
    
    target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
    
    #target = np.expand_dims(img,axis  = 3)
    
    return target

def listDataset(path, batch_size = 64):
    """
    path: array of image paths
    batch_size: no. data samples to train on
    """
    batch_path = np.random.choice(a = path, size = batch_size)
    batch_input = []
    batch_output = []
    
    for input_path in batch_paths:
        inputs = get_input(input_path)
        outputs = get_output(input_path.replace('.jpg','.h5').replace('images','ground_truth'))
        batch_input += [inputs]
        batch_output += [outputs]
        
    batch_x = np.array(batch_input)
    batch_y = np.array(batch_output)
    
    return (batch_x, batch_y)