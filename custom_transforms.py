import torch
import torchvision
import torchvision.transforms as transforms

import math
import numpy as np
from PIL import Image


class Denorm(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):    
        return tensor.mul(self.std).add(self.mean)

#bbox transforms
class ResizeImgAndBbox(object):
    def __init__(self, size_tup):
        self.size = size_tup
        
    def __call__(self, sample):
        #bbox format (top left corner, bottom right corner) [y1, x1, y2, x2]
        img = sample['image'] 
        bbox_arr = sample['bbox']
        
        w, h = img.size
        w_r, h_r = self.size
        bbox = bbox_arr.copy()
        
        bbox[0] = int(bbox_arr[0]*(h_r/h))
        bbox[2] = int(bbox_arr[2]*(h_r/h))
        
        bbox[1] = int(bbox_arr[1]*(w_r/w))
        bbox[3] = int(bbox_arr[3]*(w_r/w))
        
        return { 'image' : img.resize(self.size), 'bbox' : bbox }

class RandomFlipImgAndBbox(object):    
    def __call__(self, sample):
        img = sample['image']
        bbox = sample['bbox']
        
        if np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            #bbox format (top left corner, bottom right corner) [y1, x1, y2, x2]
            copy = int(bbox[1])
            
            bbox[1] = w - bbox[3]
            bbox[3] = w - copy
        return { 'image' : img, 'bbox' : bbox }
    
class RandomRotateImgAndBbox(object):
    def __init__(self, deg):
        self.deg = deg
    
    def __rotate__(self, origin, point, deg):
        ox, oy = origin
        px, py = point
        theta = math.radians(-deg) #-deg since we measure y,x from top left and not w/2,h/2
        
        qx = math.cos(theta)*(px - ox) - math.sin(theta)*(py - oy)
        qy = math.sin(theta)*(px - ox) + math.cos(theta)*(py - oy)
        
        qx = qx + ox
        qy = qy + oy
        return int(qx), int(qy)
    
    def __call__(self, sample):
        img = sample['image']
        bbox_arr = sample['bbox']
        
        rand_deg = np.random.randint(-1*self.deg, self.deg+1)
        img = img.rotate(rand_deg)
        
        #bbox format (top left corner, bottom right corner) [y1, x1, y2, x2]
        bbox = bbox_arr.copy()
        y1, x1, y2, x2 = bbox_arr
        w, h = img.size
        bbox[1], bbox[0] = self.__rotate__((w/2, h/2), (x1, y1), rand_deg)
        bbox[3], bbox[2] = self.__rotate__((w/2, h/2), (x2, y2), rand_deg)
        return { 'image' : img, 'bbox' : bbox }
    
class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.tfm = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        
    def __call__(self, sample):
        img = self.tfm(sample['image'])
        return { 'image' : img, 'bbox': sample['bbox'] }
    
class ImgAndBboxToTensor(object):        
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
        
    def __call__(self, sample):        
        return { 'image' : self.ToTensor(sample['image']), 'bbox' : torch.tensor(sample['bbox'], dtype=torch.float) }      