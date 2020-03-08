import numpy as np
import cv2
import os

mean = np.zeros(3, dtype=np.float32)
std = np.zeros(3, dtype=np.float32)
no_files =0
nclasses =12
normVal  =1.10

def compute_class_weights(histogram,classWeights):
    '''
    Helper function to compute the class weights
    :param histogram: distribution of class samples
    :return: None, but updates the classWeights variable
    '''
    normHist = histogram / np.sum(histogram)
    for i in range(nclasses):
        classWeights[i] = 1 / (np.log(normVal + normHist[i]))
    return classWeights
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
    masks.sort()
    imgs.sort()

    return imgs , masks
def prepocess():
    imgs , masks = make_dataset("./data/training/")
    for filename in imgs:
        if filename == '.ipynb_checkpoints':
            continue
        name = filename[:-4]            
        print("Image Processing: ",name)
        rgb_img = cv2.imread(filename)
        mean[0] += np.mean(rgb_img[:,:,0])
        mean[1] += np.mean(rgb_img[:, :, 1])
        mean[2] += np.mean(rgb_img[:, :, 2])

        std[0] += np.std(rgb_img[:, :, 0])
        std[1] += np.std(rgb_img[:, :, 1])
        std[2] += np.std(rgb_img[:, :, 2])
        no_files += 1

    mean /= no_files
    std /= no_files

    print(mean)
    print(std)
def compute_weights():
    print("compute_weights")
    global_hist = np.zeros(nclasses, dtype=np.float32)
    classWeights = np.ones(nclasses, dtype=np.float32)
    imgs , masks = make_dataset("./data/training/")
    for filename in masks:
        print(filename)
        if filename == '.ipynb_checkpoints':
            continue
        label_img = cv2.imread(filename, 0)
        hist = np.histogram(label_img, nclasses)
        global_hist += hist[0]
    return compute_class_weights(global_hist,classWeights)
if __name__ == '__main__':
    classWeights=compute_weights()
    print(classWeights)