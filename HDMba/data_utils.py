import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
from torch.utils.data import DataLoader
from metrics import *
from option import opt
from random import randrange
from osgeo import gdal
gdal.PushErrorHandler("CPLQuietErrorHandler")

BS = opt.bs
crop_size = opt.crop_size


class RESIDE_Dataset(data.Dataset):
    # def __init__(self,path,train,size=crop_size,format='.png'):
    def __init__(self, path, train, size=crop_size, format='.tif'):
        super(RESIDE_Dataset,self).__init__()
        self.size = size
        self.train = train
        self.format = format
        # G5数据
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        # G5数据
        haze = gdal.Open(self.haze_imgs[index], 0)              # 读入的是结构
        haze_array = haze.ReadAsArray().astype(np.float32)   # 读入数组 (305,512,512)
        band, width, height = haze_array.shape
        img_path = self.haze_imgs[index]
        id = os.path.basename(img_path).split('_')[0]
        clear_name = id+self.format
        clear = gdal.Open(os.path.join(self.clear_dir, clear_name))
        clear_array = clear.ReadAsArray().astype(np.float32)
        haze_array = haze_array - (np.min(haze_array)-np.min(clear_array))
        haze_array = haze_array / np.max(haze_array)
        clear_array = clear_array / np.max(clear_array)

        # 影像裁剪成块
        if isinstance(self.size, int):
            x, y = randrange(0, width - self.size + 1), randrange(0, height - self.size + 1)
            haze_array = haze_array[:, x:x+self.size, y:y+self.size]
            clear_array = clear_array[:, x:x + self.size, y:y + self.size]

        return haze_array, clear_array

    def __len__(self):
        return len(self.haze_imgs)
        # return len(self.haze_paths)


import os
pwd = os.getcwd()

path = r'F:\GF-5 dehaze'    # path to your 'data' folder
train_loader = DataLoader(dataset=RESIDE_Dataset(path+r'\train', train=True, size=crop_size), batch_size=BS, shuffle=True)
test_loader = DataLoader(dataset=RESIDE_Dataset(path+r'\test', train=False, size='whole img'), batch_size=1, shuffle=False)

x, y = next(iter(train_loader))


if __name__ == "__main__":
    pass
