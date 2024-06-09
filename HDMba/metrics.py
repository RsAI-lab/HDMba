from math import exp
import math
import numpy as np
# import imgvision as iv # python图像光谱视觉分析库-imgvision

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from numpy.linalg import norm

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel).cpu().numpy()
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel).cpu().numpy()
    # print(type(mu2))
    mu1_sq = np.power(mu1, 2)    # mul的2次方
    mu2_sq = np.power(mu2, 2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel).cpu().numpy() - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel).cpu().numpy() - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel).cpu().numpy() - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return np.mean(ssim_map)
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)    # 将输入input张量每个元素的夹紧到区间
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

def psnr(pred, gt):
    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


def UQI(O, F):
    meanO = torch.mean(O)
    meanF = torch.mean(F)
    (_, _, m, n) = np.shape(F)
    varO = torch.sqrt(torch.sum((O - meanO) ** 2) / (m * n - 1))
    varF = torch.sqrt(torch.sum((F - meanF) ** 2) / (m * n - 1))

    covOF = torch.sum((O - meanO) * (F - meanF)) / (m * n - 1)
    UQI = 4 * meanO * meanF * covOF / ((meanO ** 2 + meanF ** 2) * (varO ** 2 + varF ** 2))
    return UQI.data.cpu().numpy()


def SAM(pred, target):
    pred = torch.clamp(pred, min=0, max=1)
    target = torch.clamp(target, min=0, max=1)
    pred1 = pred[0, :, :, :].cpu()
    target1 = target[0, :, :, :].cpu()
    pred1 = np.transpose(pred1, (2, 1, 0))
    target1 = np.transpose(target1, (2, 1, 0))
    sam_rad = np.zeros((pred1.shape[0], pred1.shape[1]))
    for x in range(pred1.shape[0]):
        for y in range(pred1.shape[1]):
            tmp_pred = pred1[x, y].ravel()
            tmp_true = target1[x, y].ravel()
            cos_value = (tmp_pred.mean() / (norm(tmp_pred) * tmp_true.mean() / norm(tmp_true)))
            # print(cos_value)
            if 1.0 < cos_value:
                cos_value = 1.0
            sam_rad[x, y] = cos_value
    SAM1 = np.arccos(sam_rad)
    # SAM1 = sam1.mean() * 180 / np.pi
    return SAM1


def calc_sam(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / torch.max(img_tgt)
    img_fus = img_fus / torch.max(img_fus)
    A = torch.sqrt(torch.sum(img_tgt**2, 0))
    B = torch.sqrt(torch.sum(img_fus**2, 0))
    AB = torch.sum(img_tgt*img_fus, 0)
    sam = AB/(A*B)
    sam = torch.arccos_(sam)
    sam = torch.mean(sam)
    return sam


if __name__ == "__main__":
    pass

    # np.random.seed(10)
    # pred = np.random.rand(1, 20, 100, 100)
    # targets = np.random.rand(1, 20, 100, 100)
    # pred = torch.rand(1, 20, 100, 100)
    # targets = torch.rand(1, 20, 100, 100)
    # sam1 = calc_sam(pred, targets)
    # sam1 = SAM(pred, targets)
    # print(sam1)
    #
    # pred1 = np.transpose(pred, (2, 1, 0))
    # targets1 = np.transpose(targets, (2, 1, 0))
    # Metric = iv.spectra_metric(pred1, targets1)
    # SAM1 = Metric.SAM()
    # print(SAM1)
    #
    # UQI1 = UQI(pred, targets)
    # print(UQI1)
    #
    # pred2 = torch.Tensor(pred)
    # targets2 = torch.Tensor(targets)
    # ssim1 = ssim(pred2, targets2).item()
    # print(ssim1)
    # psnr1 = psnr(pred2, targets2)
    # print(psnr1)



