import numpy as np
import torch
import random
import cv2
import torchvision
import numbers
import torchvision.transforms.functional as F
__author__ = "Sachin Mehta"

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img, label):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img)),label
class CCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Scale(object):
    """
    Randomly crop and resize the given PIL image with a probability of 0.5
    """
    def __init__(self, wi, he):
        '''

        :param wi: width after resizing
        :param he: height after reszing
        '''
        self.w = wi
        self.h = he

    def __call__(self, img, label):
        '''
        :param img: RGB image
        :param label: semantic label image
        :return: resized images
        '''
        #bilinear interpolation for RGB image
        img = cv2.resize(img, (self.w, self.h))
        # nearest neighbour interpolation for label image
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        return [img, label]



class RandomCropResize(object):
    """
    Randomly crop and resize the given PIL image with a probability of 0.5
    """
    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between o and crop area
        '''
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h-y1, x1:w-x1]
            label_crop = label[y1:h-y1, x1:w-x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w,h), interpolation=cv2.INTER_NEAREST)
            return img_crop, label_crop
        else:
            return [img, label]

class RandomCrop(object):
    '''
    This class if for random cropping
    '''
    def __init__(self, cropArea):
        '''
        :param cropArea: amount of cropping (in pixels)
        '''
        self.crop = cropArea

    def __call__(self, img, label):

        if random.random() < 0.5:
            h, w = img.shape[:2]
            img_crop = img[self.crop:h-self.crop, self.crop:w-self.crop]
            label_crop = label[self.crop:h-self.crop, self.crop:w-self.crop]
            return img_crop, label_crop
        else:
            return [img, label]



class RandomFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, image, label):
        if random.random() < 0.5:
            x1 = 0#random.randint(0, 1) #if you want to do vertical flip, uncomment this line
            if x1 == 0:
                image = cv2.flip(image, 0) # horizontal flip
                label = cv2.flip(label, 0) # horizontal flip
            else:
                image = cv2.flip(image, 1) # veritcal flip
                label = cv2.flip(label, 1)  # veritcal flip
        return [image, label]
    
class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value
    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
            

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))
        random.shuffle(transforms)
        transform = CCompose(transforms)
        return transform
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        if random.random() < 0.25:
            transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
            return [transform(img), label]
        else : 
            return [img, label]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class RamdomNoise(object):
    def __init__(self,noise_value=30, prob=30):
        self.noise_value = noise_value
        self.prob = prob
    def __call__(self, image, label):
        if random.random() < self.prob:
            width, height = image.size
            noise = np.random.randint(-self.noise_value, self.noise_value)
            image = np.array(image, dtype= np.float32)
            image += noise
            image = np.clip(image,0,255)
            image = Image.fromarray(np.unit8(image))
        return [image , label]
    
class RGBlinearcorrection(object):
    def __init__(self,value,prob=0.3):
        self.value = value
        self.prob = prob
    def __call__(self, image, label):
        if random.random()<self.prob:
            select_nch = ranfom.randint(1,3)
            select_chs = random.sample([0,1,2], k=select_nch)
            image = np.array(image)
            for select_ch in select_chs:
                image[:,:,select_ch]=image[:,:,select_ch]*self.value
            return [Image.fromarray(np.unit8(image)),label]
        else:
            return [image, label]
class Normalize(object):
    """Given mean: (R,G,B) and std: (R,G,B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    

    """

    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        #image = image.astype(np.float32)
        img = image.copy()
        img = np.asarray(img , dtype="float32" )
        lab = label.copy()
        lab = np.asarray(lab , dtype="float32" )
        img = img/255
        for j in range(3):
            img[:, :, j] = (img[:, :, j]-self.mean[j])/self.std[j]
        '''
        for i in range(3):
            image[:,:,i] -= self.mean[i]
        for i in range(3):
            image[:,:, i] /= self.std[i]
        '''
        return [img, lab]

class ToTensor(object):
    '''
    This class converts the data to tensor so that it can be processed by PyTorch
    '''
    def __init__(self, scale=1):
        '''
        :param scale: ESPNet-C's output is 1/8th of original image size, so set this parameter accordingly
        '''
        self.scale = scale # original images are 2048 x 1024

    def __call__(self, image, label):

        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w/self.scale), int(h/self.scale)), interpolation=cv2.INTER_NEAREST)

        image = image.transpose((2,0,1))

        image_tensor = torch.from_numpy(image)
        label_tensor =  torch.LongTensor(np.array(label, dtype=np.int)) #torch.from_numpy(label)

        return [image_tensor, label_tensor]

class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
