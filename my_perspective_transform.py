import  torch
import torch.nn.functional as F 
import cv2
import numpy as np

def spatial_transformer_network(input_fmap, theta, **kwargs):
    """
    Input
    -----
    - theta: tensor of shape (B, 9)
    - input_fmap: tensor of shape (B, C, H, W)

    Returns
    -------
    - output: tensor of shape (B, C, H, W)
    """   
    # grab input dimensions
    B = input_fmap.size(0)
    H = input_fmap.size(2)
    W = input_fmap.size(3)
    # vector_ones = torch.cuda.FloatTensor(np.ones((B,1)))
    theta = torch.cuda.FloatTensor(theta)
    # theta = torch.cat([theta, vector_ones], axis = 1)
    # # reshape theta to (B, 3, 3)
    theta = torch.reshape(theta, [B, 3, 3])
    # generate grids of same size or upsample/downsample if specified
    batch_grids = affine_grid_generator(H, W, theta)
    
    batch_grids = batch_grids.permute((0,2,3,1))/H * 2 -1 # (N,H,W,2)
    # sample input with grid to get output
    out_fmap = F.grid_sample(input_fmap, batch_grids)
    return out_fmap



def affine_grid_generator(height, width, theta):
    """
    Input
    -----
    - height: H
    - width: W
    - theta: tensor of shape (B,3,3)

    Returns
    -------
    - output: tensor of shape (B, 2, H, W)
    """
    B = theta.size(0)
    x = torch.linspace(0, width-1, width)
    y = torch.linspace(0, height-1, height)
    y_t, x_t = torch.meshgrid(x, y)
    # flatten
    x_t_flat = x_t.flatten().cuda()
    y_t_flat = y_t.flatten().cuda()
    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = torch.ones_like(x_t_flat).cuda()
    sampling_grid = torch.stack([x_t_flat, y_t_flat, ones])
    # repeat grid B times
    sampling_grid = sampling_grid.unsqueeze(0)
    sampling_grid = sampling_grid.repeat(B,1,1)

    sampling_grid = torch.cuda.FloatTensor(sampling_grid) 
    # transform the sampling grid - batch multiply
    batch_grid3 = torch.matmul(theta, sampling_grid)
    # batch grid has shape (B, 3, H*W)
    batch_grids1 = torch.div(batch_grid3[:,0,:], batch_grid3[:,2,:])
    batch_grids2 = torch.div(batch_grid3[:,1,:], batch_grid3[:,2,:])
    batch_grids = torch.cat([batch_grids1, batch_grids2], axis = 1)
    # reshape to (B, H, W, 2)
    batch_grids = torch.reshape(batch_grids, [B, 2, height, width])
    return batch_grids
