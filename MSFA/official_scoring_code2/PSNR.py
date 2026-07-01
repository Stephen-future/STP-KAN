import math
import os
import hdf5storage as hdf5
import numpy as np


def computerPSNR(SR, GT ):

    # SR:重建图像  GT:原始数据图像  scale:尺寸大小
    # assume RGB image
    SR_data = np.array(SR).astype(np.float32)
    # SR_data = SR_data[scale:-scale,scale:-scale]
    GT_data = np.array(GT).astype(np.float32)
    # GT_data = GT_data[scale:-scale,scale:-scale]

    diff = GT_data - SR_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))        #
    return 20 * math.log10(1/rmse)

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-4:] == '.mat':
                ret.append(filespath)
    return ret

def folder_img_PSNR(generated_folder_path, groundtruth_folder_path):
    matlist = get_jpgs(generated_folder_path)
    avg_PSNR = 0
    psnr_sum_3ch = 0
    psnr_sum_4ch = 0
    psnr_sum_5ch = 0
    count_3ch = count_4ch = count_5ch = 0
    for i, matname in enumerate(matlist):
        generated_mat_path = os.path.join(generated_folder_path, matname)
        groundtruth_mat_path = os.path.join(groundtruth_folder_path, matname)
        generated_mat = hdf5.loadmat(generated_mat_path)['cube']
        groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube']

        # peak = 1

        psnr = computerPSNR(generated_mat, groundtruth_mat)
        # 获取通道数
        channels = generated_mat.shape[2]  # 获取通道数，假设最后一个维度是通道数
        # 根据通道数累加PSNR并统计数量
        if channels == 9:
            psnr_sum_3ch += psnr
            count_3ch += 1
        elif channels == 16:
            psnr_sum_4ch += psnr
            count_4ch += 1
        elif channels == 25:
            psnr_sum_5ch += psnr
            count_5ch += 1

        # 计算每个通道数的平均PSNR
    avg_psnr_3ch = psnr_sum_3ch / count_3ch if count_3ch > 0 else 0
    avg_psnr_4ch = psnr_sum_4ch / count_4ch if count_4ch > 0 else 0
    avg_psnr_5ch = psnr_sum_5ch / count_5ch if count_5ch > 0 else 0

    # 计算总的平均PSNR
    total_psnr = (psnr_sum_3ch + psnr_sum_4ch + psnr_sum_5ch) / (count_3ch + count_4ch + count_5ch) if (count_3ch + count_4ch + count_5ch) > 0 else 0

    # 打印输出结果
    print('9通道图像的平均PSNR:', avg_psnr_3ch)
    print('16通道图像的平均PSNR:', avg_psnr_4ch)
    print('25通道图像的平均PSNR:', avg_psnr_5ch)
    print('总的平均PSNR:', total_psnr)

    # 返回结果
    return {
        'avg_psnr_3ch': avg_psnr_3ch,
        'avg_psnr_4ch': avg_psnr_4ch,
        'avg_psnr_5ch': avg_psnr_5ch,
        'total_avg_psnr': total_psnr
    }

original_folder = r'D:\code\data\Valid_spectral_crop'
reconstructed_folder = r'C:\Users\Stephen\Desktop\RF_unet\0.5+0.3+0.2\valid_rec_results'
result = folder_img_PSNR(reconstructed_folder, original_folder)
print(result)