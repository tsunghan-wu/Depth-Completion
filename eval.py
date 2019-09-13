import os, sys
import glob
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from .utils.loss_func import pytorch_ssim

def miss(output_depth, render_depth, mask, t):
    output_over_render, render_over_output \
        = output_depth/render_depth, render_depth/output_depth
    output_over_render[np.isnan(output_over_render)] = 0
    render_over_output[np.isnan(render_over_output)] = 0
    miss_map = np.maximum(output_over_render, render_over_output)
    hit_rate = np.sum(miss_map[~mask] < t).astype(float) 
    return hit_rate

def depth_rel(output_depth, render_depth, mask):
    diff = np.abs((output_depth-render_depth))/np.abs(render_depth)
    diff[mask] = 0 # remove nan
    return diff[~mask].reshape(-1)

if __name__ == "__main__":

    output_dir = sys.argv[1]
    flist = glob.glob(os.path.join(output_dir, "*render.npy"))
    flist = [os.path.basename(x)[:-11] for x in flist]

    ssim_helper = pytorch_ssim.SSIM(11)

    REL = []
    L1 = []
    RMSE = []
    SSIM = []
    delta_105 = []
    delta_110 = []
    delta_125_1 = []
    delta_125_2 = []
    delta_125_3 = []
    N = 0
    for fname in tqdm(flist):
        gt = f'{output_dir}{fname}_render.npy'
        pd = f'{output_dir}{fname}_output.npy'

        gt_np = np.load(gt)
        pd_np = np.load(pd)

        mask = (gt_np == 0.0 )
        pd_np[mask] = 0.0
        gt_np[mask] = 0.0

        # calculate valid pixels
        n = 256 * 320 - np.sum(mask) 
        N += n
        
        # calculate hit
        delta_105.append(miss(pd_np, gt_np, mask, 1.05))
        delta_110.append(miss(pd_np, gt_np, mask, 1.10))
        delta_125_1.append(miss(pd_np, gt_np, mask, 1.25))
        delta_125_2.append(miss(pd_np, gt_np, mask, 1.25**2))
        delta_125_3.append(miss(pd_np, gt_np, mask, 1.25**3))

        # calculate mse
        RMSE.append(((gt_np-pd_np)**2))
        
        # calculate L1
        L1.append(np.abs(gt_np-pd_np))

        # calculate rel
        rel_err = depth_rel(pd_np, gt_np, mask)
        REL += (list(rel_err))

        # calculate ssim
        gt_np = gt_np.reshape(1, 1, 256, 320)
        pd_np = pd_np.reshape(1, 1, 256, 320)
        gt_tensor = torch.Tensor(gt_np)
        pd_tensor = torch.Tensor(pd_np)
        ssim = ssim_helper(gt_tensor, pd_tensor).data.numpy()
        SSIM.append(ssim)

    SSIM = np.mean(SSIM)
    RMSE = np.sqrt(np.mean(RMSE))
    L1 = np.mean(L1)
    REL = np.median(np.array(REL).reshape(-1))
    delta_105 = np.sum(delta_105) / N
    delta_110 = np.sum(delta_110) / N
    delta_125_1 = np.sum(delta_125_1) / N
    delta_125_2 = np.sum(delta_125_2) / N
    delta_125_3 = np.sum(delta_125_3) / N

    print (f'SSIM : {SSIM}')
    print (f'RMSE : {RMSE}')
    print (f'L1 : {L1}')
    print (f'REL : {REL}')
    print (f'1.05 : {delta_105}')
    print (f'1.10 : {delta_110}')
    print (f'1.25 : {delta_125_1}')
    print (f'1.25^2 : {delta_125_2}')
    print (f'1.25^3 : {delta_125_3}')

