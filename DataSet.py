import torch
import cv2
import torch.utils.data
import os
__author__ = "Sachin Mehta"

def make_dataset(root):
    imgs=[]
    masks=[]
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

    conbine=list(zip(imgs,masks))

    return conbine
class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, root, transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        x_path, y_path = self.imgs[index]
        image = cv2.imread(x_path)
        label = cv2.imread(y_path, 0)
        if self.transform:
            [image, label] = self.transform(image, label)
        return (image, label)
    def __len__(self):
        return len(self.imgs)
