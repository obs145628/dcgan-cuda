import os
import sys
import numpy as np
import tensors_saver
from PIL import Image

if os.path.isfile(sys.argv[2]): sys.exit(0)

tensors_saver.set_out_path(sys.argv[2])
SRC_DIR = sys.argv[1]
CELEB_DIR = os.path.join(SRC_DIR, 'celebA')


def get_file(path):
    im = Image.open(os.path.join(CELEB_DIR, path))
    im = im.resize((64, 64), Image.ANTIALIAS)
    im = np.array(im)
    return im


files = os.listdir(CELEB_DIR)[:10]
data = np.array([get_file(x) for x in files]).astype(np.float32) / 255
tensors_saver.add(data)
