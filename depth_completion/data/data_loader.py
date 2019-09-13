import sys, os
import random
import numpy as np
from PIL import Image
from skimage.transform import resize as numpy_resize
import warnings

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class DepthDataset(Dataset):
    def __init__(self, dataset_name, data_path, train=True):
        self.dataset_name = dataset_name
        if self.dataset_name not in {'matterport'}:
            raise Exception(f'Dataset name not found: {self.dataset_name}')
        self.data_root = data_path
        self.len = 0
        self.train = train
        self.scene_name = []
        self.color_name = []
        self.depth_name = []
        self.normal_name = []
        self.render_name = []
        self.boundary_name = []
        self.depth_boundary_name = []
        
        if self.dataset_name == 'matterport':
            self._load_data_name_matterport(train=self.train)
   
    def _load_data_name_matterport(self, 
            data_list_filename=os.path.join(os.path.dirname(__file__), 
                'data_list', 'mp_train_list_noup.txt'), 
            test_list_filename=os.path.join(os.path.dirname(__file__), 
                'data_list', 'mp_test_list_horizontal.txt'), 
            train=True):
        if train:
            data_list = self._get_data_list(data_list_filename)
        else:
            test_list = self._get_data_list(test_list_filename)
        for x in os.listdir(self.data_root):
            scene = os.path.join(self.data_root, x)
            if os.path.isdir(scene) == False or len(x) != 11:
                # not scene directory
                continue
            raw_depth_scene = os.path.join(scene, 'undistorted_depth_images')
            render_depth_scene = os.path.join(scene, 'render_depth')
            for y in os.listdir(raw_depth_scene):
                valid, resize_count, one_scene_name, num_1, num_2, png = self._split_matterport_path(y)
                if valid == False or png != 'png' or resize_count != 1:
                    continue
                data_id = (x, one_scene_name, num_1, num_2)
                if train == True:
                    if data_id not in data_list:
                        continue
                else:
                    if data_id not in test_list:
                        continue
                raw_depth_f = os.path.join(raw_depth_scene, y)
                render_depth_f = os.path.join(render_depth_scene, y.split('.')[0] + '_mesh_depth.png')
                color_f = os.path.join(scene, 'undistorted_color_images', f'resize_{one_scene_name}_i{num_1}_{num_2}.jpg')
                est_normal_f = os.path.join(scene, 'estimate_normal', f'resize_{one_scene_name}_d{num_1}_{num_2}_normal_est.png')
                est_boundary_f = os.path.join(scene, 'estimate_boundary', f'resize_{one_scene_name}_d{num_1}_{num_2}_bound_est.png')
                depth_boundary_f = os.path.join(scene, 'sobel_edge', y.split('.')[0]+'_mesh_depth.png')
        
                feature_files = [render_depth_f, color_f, est_normal_f, est_boundary_f, depth_boundary_f]
                if self._check_file(feature_files) == False:
                    print(f'file not exists:{feature_files}')
                    continue
                temp_scene_name = y.split(".")[0]
                if "resize_" in temp_scene_name:
                    temp_scene_name = temp_scene_name[7:]
                self.scene_name.append(f'{x}_{temp_scene_name}')
                self.depth_name.append(raw_depth_f)
                self.render_name.append(render_depth_f)
                self.color_name.append(color_f)
                self.normal_name.append(est_normal_f)
                self.boundary_name.append(est_boundary_f)
                self.depth_boundary_name.append(depth_boundary_f)
        self.len = len(self.depth_name)
        print(f'total {"training" if train else "testing"} data: {self.len}')
    
    def _get_data_list(self, filename):
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        data_list = []
        for ele in content:
            left, _, right = ele.split('/')
            valid, resize_count, one_scene_name, num_1, num_2, png = self._split_matterport_path(right)
            if valid == False:
                print(f'valid data_id in datalist: {ele}')
            data_list.append((left, one_scene_name, num_1, num_2))
        return set(data_list)

    def _check_file(self, filename):
        if isinstance(filename, list):
            for one_filename in filename:
                if os.path.exists(one_filename) == False:
                    return False
            return True
        else:
            return os.path.exists(filename)

    def _split_matterport_path(self, path):
        try:
            left, png = path.split('.')
            lefts = left.split('_')
            resize_count = left.count('resize')
            one_scene_name = lefts[resize_count]
            num_1 = lefts[resize_count+1][-1]
            num_2 = lefts[resize_count+2]
            return True, resize_count, one_scene_name, num_1, num_2, png
        except:
            return False, None, None, None, None, None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.dataset_name == 'matterport':
            render_depth = np.array(Image.open(self.render_name[index]))
            depth = np.array(Image.open(self.depth_name[index]))
            mask = np.zeros_like(depth)
            mask[np.where(depth > 0.0)] = 1
            color = np.array(Image.open(self.color_name[index]))
            normal = np.array(Image.open(self.normal_name[index]))
            boundary = np.expand_dims(np.array(Image.open(self.boundary_name[index]))[:,:,2], 2)
            depth_boundary = np.array(Image.open(self.depth_boundary_name[index]))
            batch = {'scene_name': self.scene_name[index], 'color' : color, 'depth' : depth, 'mask' : mask, 'render_depth' : render_depth, 'normal' : normal, 'boundary': boundary, 'depth_boundary' : depth_boundary}
        return batch

def customed_collate_fn(dataset_name):
    if dataset_name == 'matterport':
        return customed_collate_fn_matterport
    else:
        raise Exception('Not recognized dataset name')

def customed_collate_fn_matterport(batch):
    # trans_height, trans_width = 512, 640
    trans_height, trans_width = 256, 320
    tensor_transform = transforms.Compose([
        transforms.Resize((trans_height, trans_width)),
        transforms.ToTensor(),
    ])
    def numpy_transform(value):
        if value.shape[0] != trans_height or value.shape[1] != trans_width:
            value = numpy_resize(value, (trans_height, trans_width), mode='constant', anti_aliasing=False)
        value = torch.tensor(value).type(torch.float32)
        return value
    def _transform_fn(key, value):
        if key == 'depth':
            value = numpy_transform(value)
            value /= 4000.00
            value = torch.unsqueeze(value, 0)
        elif key == 'mask':
            value = numpy_transform(value)
            value = torch.unsqueeze(value, 0)
        elif key == 'render_depth':
            value = numpy_transform(value)
            value = torch.unsqueeze(value, 0)
            value /= 4000.00
        elif key == 'color':
            value = numpy_transform(value)
            value /= 255
            value = value.permute(2, 0, 1)
        elif key == 'normal':
            value = numpy_transform(value)
            value = (value - 90) / 180
            value = value.permute(2, 0, 1)
        elif key == 'boundary':
            value = numpy_transform(value)
            value = (value - 127.5) / 255
            value = value.permute(2, 0, 1)
        elif key == 'depth_boundary':
            value = numpy_transform(value)
            value /= 255
            value = torch.unsqueeze(value, 0)
        return value
            
    keys = list(batch[0].keys())
    values = {}
    for key in keys:
        if key == 'scene_name':
            values[key] = [one_batch[key] for one_batch in batch]
        else:
            this_value = torch.stack([_transform_fn(key, one_batch[key]) for one_batch in batch], 0, out=None)
            values[key] = this_value
    return values

