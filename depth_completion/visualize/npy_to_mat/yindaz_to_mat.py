import os
from PIL import Image
import numpy as np
import scipy.io as sio

from parse_filename import parse_filename

yindaz_path = '/tmp2/tsunghan/experiment_result/mat_npy/yindaz_output'
save_dir = './yindaz'

for filename in os.listdir(yindaz_path):
    path = os.path.join(yindaz_path, filename)
    img = Image.open(path)
    img = np.array(img, dtype=np.uint16)
    save_path = os.path.join(save_dir, parse_filename(path))
    sio.savemat(save_path, {'yindaz': img})    
