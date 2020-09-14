import os
import cv2
import numpy as np
import numpy.random as random
import math
import torch
import logging
import  matplotlib.pyplot as plt
import math

from torch.utils.data import Dataset
# from my_perspective_transform import spatial_transformer_network




##############################################    pattern    #######################################
def TransformPattern(pattern, args):
    #opencv-python 
    pattern = pattern.permute((0,2,3,1))
    Ms = opencv_get_rand_transform_matrix(args.image_height, args.image_width, args.pattern_transformed_scope, args.batch_size)
    transformed_pattern = image_perspective_transform(pattern, Ms, args)
    warped_mask = image_perspective_transform(torch.ones_like(pattern), Ms, args)
    transformed_pattern += (1-warped_mask)
    transformed_pattern = transformed_pattern.permute((0,3,1,2))
    return transformed_pattern
def GammaAdjustPattern(pattern, args):
    gamma = random.uniform(1, args.pattern_gamma_adjust_scope, (args.batch_size, 1, 1, 1))
    gamma = torch.cuda.FloatTensor(gamma)
    pattern = pattern**gamma
    # pattern = torch.clamp(pattern, 0, 1)
    return pattern 
def ContrastPattern(pattern, args):
    contrast_quality = random.uniform(1-args.pattern_contrast_scope, 1+args.pattern_contrast_scope, (args.batch_size, 1, 1, 1))
    contrast_quality = torch.cuda.FloatTensor(contrast_quality)
    pattern = contrast_quality*pattern
    pattern = torch.clamp(pattern, 0, 1)
    return pattern 
def ShiftPatternBrightness(pattern, args):
    shift = random.uniform(-args.pattern_brightness_shift_scope, args.pattern_brightness_shift_scope, (args.batch_size, 1, 1, 1))
    shift = torch.cuda.FloatTensor(shift)
    pattern += shift
    pattern = torch.clamp(pattern, 0, 1)
    return pattern
def ShiftPatternHue(pattern, args):
    shift = random.uniform(-args.pattern_hue_shift_scope, args.pattern_hue_shift_scope, (args.batch_size, 3, 1, 1))
    shift = torch.cuda.FloatTensor(shift)
    pattern = pattern + shift
    pattern = torch.clamp(pattern, 0, 1)
    return pattern

##############################################    cover    #######################################
def ShiftCoverBrightness(cover, args):
    shift = random.uniform(-args.cover_brightness_shift_scope, args.cover_brightness_shift_scope, (args.batch_size, 1, 1, 1))
    shift = torch.cuda.FloatTensor(shift)
    cover += shift
    cover = torch.clamp(cover, 0, 1)
    return cover
def ContrastCover(cover, args):
    contrast_quality = random.uniform(1-args.cover_contrast_scope, 1+args.cover_contrast_scope, (args.batch_size, 1, 1, 1))
    contrast_quality = torch.cuda.FloatTensor(contrast_quality)
    cover = cover * contrast_quality
    cover = torch.clamp(cover, 0, 1)
    return cover 
def ShiftCoverHue(cover, args):
    shift = random.uniform(-args.cover_hue_shift_scope, args.cover_hue_shift_scope, (args.batch_size, 3, 1, 1))
    shift = torch.cuda.FloatTensor(shift)
    cover = cover + shift
    cover = torch.clamp(cover, 0, 1)
    return cover


##############################################    stegaimage    #######################################
def TransformImage(image, args):
    #opencv-python 
    image = image.permute((0,2,3,1))
    Ms = opencv_get_rand_transform_matrix(args.image_height, args.image_width, args.image_transformed_scope, args.batch_size)
    transformed_image = image_perspective_transform(image, Ms, args)
    warped_mask = image_perspective_transform(torch.ones_like(image), Ms, args)
    transformed_image += (1-warped_mask)
    transformed_image = transformed_image.permute((0,3,1,2))
    return transformed_image
def ShiftImageBrightness(image, args):
    shift = random.uniform(-args.image_brightness_shift_scope, args.image_brightness_shift_scope, (args.batch_size, 1, 1, 1))
    shift = torch.cuda.FloatTensor(shift)
    image += shift
    image = torch.clamp(image, 0, 1)
    return image
def ContrastImage(image, args):
    contrast_quality = random.uniform(1-args.image_contrast_scope, 1+args.image_contrast_scope, (args.batch_size, 1, 1, 1))
    contrast_quality = torch.cuda.FloatTensor(contrast_quality)
    image = image * contrast_quality
    image = torch.clamp(image, 0, 1)
    return image 
def ShiftImageHue(image, args):
    shift = random.uniform(-args.image_hue_shift_scope, args.image_hue_shift_scope, (args.batch_size, 3, 1, 1))
    shift = torch.cuda.FloatTensor(shift)
    image = image + shift
    image = torch.clamp(image, 0, 1)
    return image
def ImageNoise(image, args):
    sigma = random.uniform(0, args.image_noise_scope, (args.batch_size, 1, 1, 1))
    noise = random.normal(0, sigma)
    noise = torch.cuda.FloatTensor(noise)
    image += noise
    image = torch.clamp(image, 0, 1)
    return image
def GammaAdjustImage(image, args):
    gamma = random.uniform(args.image_gamma_adjust_scope, 1, (args.batch_size, 1, 1, 1))
    gamma = torch.cuda.FloatTensor(gamma)
    image = image**gamma
    # image = torch.clamp(image, 0, 1)
    return image
def GenerateBlurKernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur
    coords = torch.cuda.FloatTensor(torch.stack(torch.meshgrid(torch.arange(0, N_blur, dtype = torch.float32), torch.arange( 0, N_blur, dtype = torch.float32)), -1)) - (.5 * (N-1))
    # coords = tf.to_float(coords)
    manhat = torch.sum(torch.abs(coords), 2)

    vals_nothing = torch.cuda.FloatTensor(manhat )

    # gauss

    sig_gauss = torch.rand([]) * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
    vals_gauss = torch.exp(-torch.sum(coords**2, -1)/2./sig_gauss**2)

    # line

    theta = torch.rand([]) * 2.*np.pi
    v = torch.cuda.FloatTensor([torch.cos(theta), torch.sin(theta)])
    dists = torch.sum(coords * v, -1)

    sig_line = torch.rand([]) * (sigrange_line[1]-sigrange_line[0]) + sigrange_line[0]
    w_line = torch.rand([])*(.5 * (N-1) + .1 - wmin_line) + wmin_line

    vals_line = torch.exp(-dists**2/2./sig_line**2) * torch.cuda.FloatTensor(manhat)

    t = torch.rand([])
    vals = vals_nothing
    if  t < probs[0]+probs[1]:
        vals = vals_line
    if t < probs[0]:
        vals = vals_gauss
    v = vals / torch.sum(vals)
    z = torch.zeros_like(v)
    f = torch.reshape(torch.stack([v,z,z,z,v,z,z,z,v]), [3,3,N,N])
    # print(sum(v))

    return f
def JPEGCompression(image, args):
    image = image*255
    image = np.uint8(image.permute((0,2,3,1)).numpy())
    for i in range(args.batch_size):
        jpeg_quality = random.uniform(args.jpeg_quality, 100)
        cv2.imwrite(args.example_image_path + str(i) + '_JPEG.jpg', image[i,:,:,:], [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        # cv2.imwrite( './Allnoise/JPEG.jpg', image[i,:,:,:])
        single_image = cv2.imread(args.example_image_path + str(i) + '_JPEG.jpg')
        image[i,:,:,:] = single_image
    image = torch.cuda.FloatTensor(np.float32(image))
    image = image.permute((0,3,1,2))
    image = image/255
    return image

def opencv_get_rand_transform_matrix(image_height,image_width, d, batch_size):
    Ms = np.zeros((batch_size, 2, 3, 3))

    for i in range(batch_size):
        tl_x = random.uniform(-d, d)     # Top left corner, top
        tl_y = random.uniform(-d, d)    # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)    # Bot left corner, left
        tr_x = random.uniform(-d, d)     # Top right corner, top
        tr_y = random.uniform(-d, d)   # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)   # Bot right corner, right

        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_height, tr_y],
            [br_x + image_height, br_y + image_width],
            [bl_x, bl_y +  image_width]], dtype = "float32")

        dst = np.array([
            [0, 0],
            [image_height, 0],
            [image_height, image_width],
            [0, image_width]], dtype = "float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
        Ms[i,0,:,:] = M
        Ms[i,1,:,:] = M_inv
    return Ms

def image_perspective_transform(image, Ms, args):
    transformed_image = torch.ones_like(image)
    image = image.numpy()
    transformed_image = transformed_image.numpy()
    for i in range(args.batch_size):
        transformed_single_image=cv2.warpPerspective(image[i,:,:,:],Ms[i,:,:],(args.image_height,args.image_width))
        transformed_image[i,:,:,:] = transformed_single_image
    transformed_image = torch.from_numpy(transformed_image)
    return transformed_image





def SaveExampleImage(x, tag, args):
    # # 保存示例图片
    temp = x[0,:,:,:]
    if temp.device == 'cpu':
        temp = np.uint8(temp.permute((1,2,0)).numpy() * 255)
    else:
        temp = np.uint8(temp.permute((1,2,0)).cpu().detach().numpy() * 255)
    
    cv2.imwrite(args.example_image_path + str(args.global_epoch) + tag + '.png', temp)


def UpdateGloabalSummary(summary:dict, global_summary:dict, args):
    for key,value in summary.items():
        if key != 'step' and key != 'epoch':
            
            global_summary.setdefault(key,[]).append(value)
            if args.global_epoch != 0:
                PlotCurve(global_summary[key], key, args)
    return global_summary

 # 绘制曲线
def PlotCurve(y, tag, args):
    # x list    
    x = range(args.global_epoch)
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel(tag)
    plt.xlim(0, args.global_epoch)
    y_max = math.floor(max(y)) + 1
    plt.ylim(0, y_max)
    plt.title(tag + '-epoch')
    plt.savefig(args.record_path + tag + '.png')
    plt.close()

def GetLog(file_name): #设置日志
    logger = logging.getLogger('train')  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级
 
    ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.INFO)  # 设定输出hander的level

    fh = logging.FileHandler(file_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever
 
 
 
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
    logger.addHandler(ch)
    return logger


