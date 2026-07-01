import os
import re
import random
import numpy as np
import cv2
# import scipy.io as io
import torch
from torch.utils.data import Dataset
from PIL import Image
import utils
import hdf5storage as hdf5
# import scipy.ndimage
# import scipy.signal

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)         # 只能写3 或 4
    # 若是3，(256, 256, 3)；若是4，(256, 256, 4)
    return out_np

def load_CFAmat(img_path):
    out_np = np.asarray(hdf5.loadmat(img_path)['mosaic'])        # 480*512  uint16
    # out_np = np.asarray(hdf5.loadmat(img_path)['I_MOS'])
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)         # 只能写3 或 4
    # 若是3，(256, 256, 3)；若是4，(256, 256, 4)
    return out_np
# ###########=============================
def gasuss_noise(SSI, mean=0, var_max=5):
    ''' SSI 是输入的快照光谱数据，[0,1]
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    noise = np.random.uniform(0,var_max,1)* np.random.normal(mean, 1,SSI.shape)/4095
    out = SSI + noise
     #cv.imshow("gasuss", out)
    return out

# 多尺度数据集
class HS_multiscale_DSet(Dataset):
    def __init__(self, opt):                                   		    # root: list ; transform: torch transform
        self.opt = opt
        # the root of both domains
        # self.baseroot_A = os.path.join(opt.baseroot,opt.train_data_reference)             # HS
        # self.baseroot_B = os.path.join(opt.baseroot,opt.train_data_in)             # CFA
        self.baseroot_A = opt.train_data_reference
        self.baseroot_B = opt.train_data_in
        self.baseroot_C = opt.train_data_reference_2
        self.baseroot_D = opt.train_data_in_2
        namelist = self.get_names(self.baseroot_A)
        namelist_2 = self.get_names(self.baseroot_C)
        # build image list
        self.imglist_A, self.imglist_B = self.build_imglist(namelist, opt)
        self.imglist_C, self.imglist_D = self.build_imglist(namelist_2, opt)

    # 数据集里的图像名长度不一样
    # ARAD_1K_0021_16.mat  和    ARAD_1K_0021.raw.mat
    def get_names(self, path):
        # read a folder, return the image name    读取文件夹，返回图像名称
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(filespath.split('\\')[-1][:-4])    # 去掉  XXXXXXXX.mat中的.mat,保留剩下的 XXXXXXXX
        ret.sort(key=lambda f:int(re.search(r'\d{4}',f).group()))
        return ret

    def build_imglist(self, namelist, opt):
        # build an imglist
        imglist_A = []
        imglist_B = []
        for i in range(len(namelist)):
            imglist_A.append(namelist[i] + '.mat')             # ARAD_1K_0021_16.mat
            if opt.save_path == 'track2':
                imglist_B.append(namelist[i] + '.mat')           # ARAD_1K_0021_16.mat    [0，4095]
        return imglist_A, imglist_B



    def __getitem__(self, index):
        # read an image
        imgpath_A = os.path.join(self.baseroot_A, self.imglist_A[index])
        img_A = hdf5.loadmat(imgpath_A)['cube']            # (480, 512, 16), in range [0, 1], double
        imgpath_B = os.path.join(self.baseroot_B, self.imglist_B[index])
        mosaic_16 = hdf5.loadmat(imgpath_B)['cube']          # (480, 512, 16), in range [0, 4095],double  mosaic

        imgpath_C = os.path.join(self.baseroot_C, self.imglist_C[index])
        img_C = hdf5.loadmat(imgpath_C)['cube']
        imgpath_D = os.path.join(self.baseroot_D, self.imglist_D[index])
        mosaic_9 = hdf5.loadmat(imgpath_D)['cube']

        if self.opt.augment:
            if np.random.uniform() < 0.5:
                img_A = np.fliplr(img_A)
                mosaic_16 = np.fliplr(mosaic_16)
                img_C = np.fliplr(img_C)
                mosaic_9 = np.fliplr(mosaic_9)
            if np.random.uniform() < 0.5:
                img_A = np.flipud(img_A)
                mosaic_16 = np.flipud(mosaic_16)
                img_C = np.flipud(img_C)
                mosaic_9 = np.flipud(mosaic_9)
            k = random.randint(1, 4)
            img_A = np.rot90(img_A, k)
            mosaic_16 = np.rot90(mosaic_16, k)
            img_C = np.rot90(img_C, k)
            mosaic_9 = np.rot90(mosaic_9, k)
        # crop   随机裁剪
        if self.opt.crop_size > 0:
            h, w = img_A.shape[:2]      # 取高和宽
            # rand_h = int(random.randint(0, h - self.opt.crop_size)/4)*4
            # rand_w = int(random.randint(0, w - self.opt.crop_size)/4)*4
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            img_A = img_A[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]  #(256, 256, 16)
            mosaic_16 = mosaic_16[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]

            img_C = img_C[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size, :]
            mosaic_9 = mosaic_9[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size, :]

        img_A = torch.from_numpy(img_A.astype(np.float32).transpose(2, 0, 1)).contiguous()    # HS.mat      16*480*512
        mosaic_16 = torch.from_numpy(mosaic_16.astype(np.float32).transpose(2, 0, 1)).contiguous()    # CFA.mat     16*480*512

        img_C = torch.from_numpy(img_C.astype(np.float32).transpose(2, 0, 1)).contiguous()
        mosaic_9 = torch.from_numpy(mosaic_9.astype(np.float32).transpose(2, 0, 1)).contiguous()


        return mosaic_16, img_A, mosaic_9, img_C



    def __len__(self):
        return len(self.imglist_A)


# 多尺度验证数据集     在validation1.py中使用
class HS_multiscale_ValDSet(Dataset):
    def __init__(self, opt):                                   		    # root: list ; transform: torch transform
        self.opt = opt
        # the root of both domains
        self.baseroot = os.path.join(opt.val_data_in)
        self.baseroot_2 = os.path.join(opt.val_data_in_2)
        # build image list
        self.imglist = utils.get_jpgs(self.baseroot)
        self.imglist_2 = utils.get_jpgs(self.baseroot_2)

    def __getitem__(self, index):
        # read an image
        imgname = self.imglist[index]
        imgpath = os.path.join(self.baseroot, imgname)
        img = hdf5.loadmat(imgpath)['cube']          # (480, 512, 16), in range [0, 4095],double  mosaic

        img1 = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1 )).contiguous()      # torch.Size([16 , 256, 256]),in range [0, 1]

        imgname_2 = self.imglist_2[index]
        imgpath_2 = os.path.join(self.baseroot_2, imgname_2)
        img_2 = hdf5.loadmat(imgpath_2)['cube']  # (480, 512, 16), in range [0, 4095],double  mosaic

        img2 = torch.from_numpy(img_2.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return img1, imgname, img2, imgname_2


    def __len__(self):
        return len(self.imglist)             # self.imglist = utils.get_jpgs(self.baseroot)
