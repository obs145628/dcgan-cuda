import glob
import os
import utils
import scipy.misc

PATH_IN = '../celebA'
PATH_OUT = '../celeba_norm'

os.makedirs(PATH_OUT, exist_ok=True)

data_files = glob.glob(os.path.join(PATH_IN, "*.jpg"))


for pin in data_files:
    pout = pin.replace('celebA', 'celeba_norm')
    f = scipy.misc.imread(pin)
    f = utils.center_crop(f, 108, 64)
    scipy.misc.imsave(pout, f)
    print(pout)


print('Done')
