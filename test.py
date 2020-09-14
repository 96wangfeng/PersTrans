import os
import cv2
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import torch
from PIL import Image,ImageOps
import torch.nn as nn
import torch.utils.data as dataf
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchgeometry

import utils 
from my_perspective_transform import spatial_transformer_network
print(torch.cuda.is_available())

original_image = cv2.imread('test_im_hidden.png')
original_image = torch.cuda.FloatTensor(original_image)
original_image = original_image.unsqueeze(0)

Ms = utils.opencv_get_rand_transform_matrix(400, 400, 50, 1)
print(Ms)
Ms = torch.cuda.FloatTensor(Ms)
original_image = original_image.permute(0,3,1,2)
transform_image = torchgeometry.warp_perspective(original_image, Ms[:,1,:,:], dsize=(400, 400), flags='bilinear')
transform_image = transform_image.permute(0,2,3,1)
print(transform_image.size())
transform_image = transform_image.cpu().detach().numpy()
transform_image = transform_image.astype(np.uint8)
transform_image = transform_image[0,:,:,:]
cv2.imwrite('transform1.png', transform_image)

original_image = cv2.imread('transform1.png')
original_image = torch.cuda.FloatTensor(original_image)
original_image = original_image.unsqueeze(0)
original_image = original_image.permute(0,3,1,2)
transform_image = spatial_transformer_network(original_image, Ms[:,1,:,:])
transform_image = transform_image.permute(0,2,3,1)
transform_image = transform_image.cpu().detach().numpy()
transform_image = transform_image.astype(np.uint8)
transform_image = transform_image[0,:,:,:]
cv2.imwrite('transform2.png', transform_image)

print(torch.matmul(Ms[0,0,:,:], Ms[0,1,:,:]))