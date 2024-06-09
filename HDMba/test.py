import argparse
import numpy as np
from models import *
import cv2
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from osgeo import gdal
from util import fuse_images, tensor2im
gdal.PushErrorHandler("CPLQuietErrorHandler")


from models.AACNet import AACNet
from models.AIDTransformer import AIDTransformer
from models.Dehazeformer import DehazeFormer
from models.HDMba import HDMba


import time
abs = os.getcwd()+'/'


def TwoPercentLinear(image, max_out=255, min_out=0):  # 2%的线性拉伸
    b, g, r = cv2.split(image)  # 分开三个波段

    def gray_process(gray, maxout = max_out, minout = min_out):
        high_value = np.percentile(gray, 98)  # 取得98%直方图处对应灰度
        low_value = np.percentile(gray, 2)    # 同理
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        processed_gray = ((truncated_gray - low_value)/(high_value - low_value)) * (maxout - minout)#线性拉伸嘛
        return processed_gray
    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = cv2.merge((b_p, g_p, r_p)) #合并处理后的三个波段
    return np.uint8(result)


def get_write_picture_fina(img):  # get_write_picture函数得到训练过程中的可视化结果
    img = tensor2im(img, np.float)
    img = img.astype(np.uint8)
    output = TwoPercentLinear(img[:, :, (58, 38, 20)])
    # output = TwoPercentLinear(img[:, :, (2, 1, 0)])
    return output


def tensorShow(name, tensors, titles):
    fig = plt.figure(figsize=(8, 8))
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        npimg = npimg / np.max(npimg)
        npimg = np.clip(npimg, 0, 1)
        ax = fig.add_subplot(121+i)
        ax.imshow(npimg)
        plt.imsave(f"../pred_GF5/{name}_{tit}.png", npimg)
        ax.set_title(tit)
    plt.tight_layout()
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--test_imgs', type=str, default='F:/L8MSI/Paired/Test/pre', help='Test imgs folder')
opt = parser.parse_args()
img_dir = opt.test_imgs+'/'

# 训练好的网络
model_dir = abs+f'trained_models/train_HDMba.pk'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir, map_location=device)
net = HDMba()

net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net = net.module.to(torch.device('cpu'))
net.eval()


for im in os.listdir(img_dir):
    # print(im)
    start_time = time.time()
    print(f'\r {im}', end='', flush=True)
    haze = gdal.Open(img_dir+im).ReadAsArray().astype(np.float32)
    haze = haze / np.max(haze)
    # haze = haze[]
    haze = np.expand_dims(haze, 0)
    haze = torch.from_numpy(haze).type(torch.FloatTensor)
    with torch.no_grad():
        pred = net(haze)
    ts = torch.squeeze(pred.cpu())
    print(f'|time_used :{(time.time() - start_time):.4f}s ')

    write_image2 = get_write_picture_fina(pred)
    write_image_name = "C:/Users/Administrator/Desktop/canet/" + str(im) + "_new.png"
    Image.fromarray(np.uint8(write_image2)).save(write_image_name)
