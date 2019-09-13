import os
from PIL import Image
import numpy as np
import scipy.io as sio

from parse_filename import parse_filename

sa_ssim_path = '/tmp2/tsunghan/experiment_result/visualize/render_depth'
save_dir = './render_depth'
if os.path.isdir(save_dir) == False:
    os.mkdir(save_dir)

for filename in os.listdir(sa_ssim_path):
    if filename.split('.')[-1] != 'png':
        continue
    path = os.path.join(sa_ssim_path, filename)
    img = np.array(Image.open(path))
    img = img.astype(np.uint16)
    save_path = os.path.join(save_dir, parse_filename(path))
    sio.savemat(save_path, {'render_depth': img})    
