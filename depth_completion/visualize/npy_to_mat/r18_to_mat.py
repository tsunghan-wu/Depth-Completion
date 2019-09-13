import os
from PIL import Image
import numpy as np
import scipy.io as sio

from parse_filename import parse_filename

sa_ssim_path = '/tmp2/tsunghan/experiment_result/mat_npy/r18sc_epo13'
save_dir = './mat/resnet'
if os.path.isdir(save_dir) == False:
    os.mkdir(save_dir)

for filename in os.listdir(sa_ssim_path):
    if '_output.npy' not in filename:
        continue
    path = os.path.join(sa_ssim_path, filename)
    img = np.load(path) * 4000 
    img = img.astype(np.uint16)
    save_path = os.path.join(save_dir, parse_filename(path))
    sio.savemat(save_path, {'resnet': img})    
