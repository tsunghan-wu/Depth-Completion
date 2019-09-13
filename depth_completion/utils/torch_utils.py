import torch
import torch.nn.functional as F


def img_grad(x):
    hg = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    vg = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    img_grad = torch.pow(hg[:,:,:,:-1] + vg[:,:,:-1,:], 0.5)
    return img_grad

def sobel_conv(x):
    # sobel filter
    sobel_x = torch.tensor([[+1, 0, -1], [+2, 0, -2], [+1, 0, -1]], requires_grad=False,dtype = torch.float)
    sobel_y = torch.tensor([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]], requires_grad=False,dtype = torch.float)
    gpu_id = x.get_device()
    sobel_x, sobel_y = sobel_x.to(gpu_id), sobel_y.to(gpu_id)
    sobel_x = sobel_x.view((1,1,3,3))
    sobel_y = sobel_y.view((1,1,3,3))
    #gradients in the x and y direction for both predictions and the target transparencies
    G_x_pred = F.conv2d(x,sobel_x,padding = 1)
    G_y_pred = F.conv2d(x,sobel_y,padding = 1)
    #magnitudes of the gradients
    M_pred = torch.pow(G_x_pred,2)+torch.pow(G_y_pred,2)
    #taking care of nans
    M_pred = (M_pred==0.).float() + M_pred
    return M_pred
