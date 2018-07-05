from random import shuffle
import scipy.misc
import numpy as np

def center_crop(x, crop_size, resize_w):
    h, w = x.shape[:2]
    j = int(round((h - crop_size)/2.))
    i = int(round((w - crop_size)/2.))
    return scipy.misc.imresize(x[j:j+crop_size, i:i+crop_size],
                               [resize_w, resize_w])

def merge(images, size):
     h, w = images.shape[1], images.shape[2]
     img = np.zeros((h * size[0], w * size[1], 3))
     for idx, image in enumerate(images):
         i = idx % size[1]
         j = idx // size[1]
         img[j*h:j*h+h, i*w:i*w+w, :] = image
     return img

def transform(image):
    return np.array(image).astype(np.float32) /127.5 - 1.

def inverse_transform(images):
     return (images+1.)/2.

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
     return scipy.misc.imsave(path, merge(images, size))

def get_image(image_path):
    return transform(imread(image_path))

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)
