"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import os
import cv2
from random import randrange
import torch.nn.functional as F


def find_patches(img_c, patch_size):
    _, band, width, height = img_c.shape
    x1, y1 = randrange(0, width - patch_size + 1), randrange(0, height - patch_size + 1)
    img_patch = img_c[:, :, x1:x1 + patch_size, y1:y1 + patch_size]
    for id in range(5):
        x, y = randrange(0, width - patch_size + 1), randrange(0, height - patch_size + 1)
        patch = img_c[:, :, x:x + patch_size, y:y + patch_size]
        img_patch = torch.cat((img_patch, patch), dim=1)
        # img_patch.append(patch)
    return img_patch.cuda()


def fuse_images(real_I, rec_J, refine_J):
    """
    real_I, rec_J, and refine_J: Images with shape hxwx3
    """
    # realness features
    mat_RGB2YMN = np.array([[0.299,0.587,0.114],
                            [0.30,0.04,-0.35],
                            [0.34,-0.6,0.17]])

    recH,recW,recChl = rec_J.shape
    rec_J_flat = rec_J.reshape([recH*recW,recChl])
    rec_J_flat_YMN = (mat_RGB2YMN.dot(rec_J_flat.T)).T
    rec_J_YMN = rec_J_flat_YMN.reshape(rec_J.shape)

    refine_J_flat = refine_J.reshape([recH*recW,recChl])
    refine_J_flat_YMN = (mat_RGB2YMN.dot(refine_J_flat.T)).T
    refine_J_YMN = refine_J_flat_YMN.reshape(refine_J.shape)

    real_I_flat = real_I.reshape([recH*recW,recChl])
    real_I_flat_YMN = (mat_RGB2YMN.dot(real_I_flat.T)).T
    real_I_YMN = real_I_flat_YMN.reshape(real_I.shape)

    # gradient features
    rec_Gx = cv2.Sobel(rec_J_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    rec_Gy = cv2.Sobel(rec_J_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    rec_GM = np.sqrt(rec_Gx**2 + rec_Gy**2)

    refine_Gx = cv2.Sobel(refine_J_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    refine_Gy = cv2.Sobel(refine_J_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    refine_GM = np.sqrt(refine_Gx**2 + refine_Gy**2)

    real_Gx = cv2.Sobel(real_I_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    real_Gy = cv2.Sobel(real_I_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    real_GM = np.sqrt(real_Gx**2 + real_Gy**2)

    # similarity
    rec_S_V = (2*real_GM*rec_GM+160)/(real_GM**2+rec_GM**2+160)
    rec_S_M = (2*rec_J_YMN[:,:,1]*real_I_YMN[:,:,1]+130)/(rec_J_YMN[:,:,1]**2+real_I_YMN[:,:,1]**2+130)
    rec_S_N = (2*rec_J_YMN[:,:,2]*real_I_YMN[:,:,2]+130)/(rec_J_YMN[:,:,2]**2+real_I_YMN[:,:,2]**2+130)
    rec_S_R = (rec_S_M*rec_S_N).reshape([recH,recW])

    refine_S_V = (2*real_GM*refine_GM+160)/(real_GM**2+refine_GM**2+160)
    refine_S_M = (2*refine_J_YMN[:,:,1]*real_I_YMN[:,:,1]+130)/(refine_J_YMN[:,:,1]**2+real_I_YMN[:,:,1]**2+130)
    refine_S_N = (2*refine_J_YMN[:,:,2]*real_I_YMN[:,:,2]+130)/(refine_J_YMN[:,:,2]**2+real_I_YMN[:,:,2]**2+130)
    refine_S_R = (refine_S_M*refine_S_N).reshape([recH,recW])

    rec_S = rec_S_R*np.power(rec_S_V, 0.4)
    refine_S = refine_S_R*np.power(refine_S_V, 0.4)

    fuseWeight = np.exp(rec_S)/(np.exp(rec_S)+np.exp(refine_S))
    return fuseWeight


def get_tensor_dark_channel(img, neighborhood_size):
    shape = img.shape
    if len(shape) == 4:
        img_min = torch.min(img, dim=1)
        img_dark = F.max_pool2d(img_min, kernel_size=neighborhood_size, stride=1)
    else:
        raise NotImplementedError('get_tensor_dark_channel is only for 4-d tensor [N*C*H*W]')

    return img_dark


def array2Tensor(in_array, gpu_id=-1):
    in_shape = in_array.shape
    if len(in_shape) == 2:
        in_array = in_array[:,:,np.newaxis]

    arr_tmp = in_array.transpose([2,0,1])
    arr_tmp = arr_tmp[np.newaxis,:]

    if gpu_id >= 0:
        return torch.tensor(arr_tmp.astype(np.float)).to(gpu_id)
    else:
        return torch.tensor(arr_tmp.astype(np.float))


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        # 计算张量的最小值和最大值
        min_val, max_val = image_numpy.min(), image_numpy.max()
        # 执行归一化操作
        image_numpy = (image_numpy - min_val) / (max_val - min_val)
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def rescale_tensor(input_tensor):
    """"Converts a Tensor array into the Tensor array whose data are identical to the image's.
    [height, width] not [width, height]

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """

    if isinstance(input_tensor, torch.Tensor):
        input_tmp = input_tensor.cpu().float()
        output_tmp = (input_tmp + 1) / 2.0 * 255.0
        output_tmp = output_tmp.to(torch.uint8)
    else:
        return input_tensor

    return output_tmp.to(torch.float32) / 255.0




