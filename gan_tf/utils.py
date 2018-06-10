from random import shuffle
import scipy.misc
import numpy as np

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def merge(images, size):
     h, w = images.shape[1], images.shape[2]
     img = np.zeros((h * size[0], w * size[1], 3))
     for idx, image in enumerate(images):
         i = idx % size[1]
         j = idx // size[1]
         img[j*h:j*h+h, i*w:i*w+w, :] = image
     return img

def transform(image, npx=64, resize_w=64):
    cropped_image = center_crop(image, npx, resize_w=resize_w)
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
     return (images+1.)/2.

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
     return scipy.misc.imsave(path, merge(images, size))

def get_image(image_path, image_size, resize_w):
    return transform(imread(image_path), image_size, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)
