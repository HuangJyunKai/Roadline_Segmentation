import torch
import cv2
import torch.utils.data
import os
import PIL.Image as Image
import numpy as np
__author__ = "Sachin Mehta"

def make_dataset(root):
    imgs=[]
    masks=[]
    #trueimg=[]
    #truemask=[]
    for filename in os.listdir(root):
        if filename == 'lineimage' :
            for filename2 in os.listdir(root+filename):
                if '.jpg' in filename2:
                    img=os.path.join(root+'lineimage/'+filename2)
                    imgs.append(img)
                    #trueimg.append(filename2[:-4])
        if filename == 'linelabel' :
            for filename2 in os.listdir(root+filename):
                if '.png' in filename2:
                    mask=os.path.join(root+'linelabel/'+filename2)
                    masks.append(mask)
                    #truemask.append(filename2[:-14])
    masks.sort()
    imgs.sort()
    '''
    trueimg.sort()
    truemask.sort()
    for i in range(len(imgs)):
        if truemask[i]!= trueimg[i]:
            print("ERROR")
    '''
    conbine=list(zip(imgs,masks))

    return conbine
class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, root, transform=None, transform_ori=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.transform_ori = transform_ori

    def __getitem__(self, index):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        x_path, y_path = self.imgs[index]
        #image = cv2.imread(x_path)
        #label = cv2.imread(y_path, 0)
        image = Image.open(x_path).convert('RGB')
        label = Image.open(y_path).convert('L')
        orignal_image = image.copy()
        orignal_label = label.copy()
        
        if self.transform:
            [aug_image, label] = self.transform(image, label)
        if self.transform_ori:
            [orignal_image,_] = self.transform_ori(orignal_image, orignal_label)
        return aug_image, label, orignal_image
    def __len__(self):
        return len(self.imgs)
def collate_fn(batch):
    images = list()
    labels = list()
    for i in batch:
        images.append(i[0])
        images.append(i[2])
        
        labels.append(i[1])
        labels.append(i[1])
    images = torch.stack(images,dim=0)
    labels = torch.stack(labels)
    return images , labels