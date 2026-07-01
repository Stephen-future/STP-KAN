import argparse
import os
import cv2
import numpy as np
import hdf5storage as hdf5
from official_scoring_code2.EvalMetrics import computeMRAE,computeRMSE,computerPSNR ,computerSAM
# from official_scoring_code2.EvalMetrics import computeMRAE,compute_rmse,compute_psnr,compute_sam
# from NTIRE2022Util import computeMRAE,compute_rmse,compute_psnr,compute_sam


BIT_8 = 256

# read path
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-4:] == '.mat':
                ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-4:] == '.mat':
                ret.append(filespath)
    return ret

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

#########========================================
## MRAE，
def folder_img_mrae(generated_folder_path, groundtruth_folder_path):

    matlist = get_jpgs(generated_folder_path)
    avg_mrae = 0
    for i, matname in enumerate(matlist):
        generated_mat_path = os.path.join(generated_folder_path, matname)
        groundtruth_mat_path = os.path.join(groundtruth_folder_path, matname)
        generated_mat = hdf5.loadmat(generated_mat_path)['cube']          # shape: (482, 512, 16)
        groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube']      # shape: (482, 512, 16)
        mrae = computeMRAE(generated_mat, groundtruth_mat)
        avg_mrae = avg_mrae + mrae
        # print('The %d-th mat\'s mrae:' % (i + 1), mrae)
    avg_mrae = avg_mrae / len(matlist)
    print('The average mrae is:', avg_mrae)
    return avg_mrae

## RMSE
def folder_img_rmse(generated_folder_path, groundtruth_folder_path):
    matlist = get_jpgs(generated_folder_path)
    avg_rmse = 0
    for i, matname in enumerate(matlist):
        generated_mat_path = os.path.join(generated_folder_path, matname)
        groundtruth_mat_path = os.path.join(groundtruth_folder_path, matname)
        generated_mat = hdf5.loadmat(generated_mat_path)['cube'] # shape: (482, 512, 16)
        groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube'] # shape: (482, 512, 16)
        rmse = computeRMSE(generated_mat, groundtruth_mat)
        avg_rmse = avg_rmse + rmse
        # print('The %d-th mat\'s rmse:' % (i + 1), rmse)
    avg_rmse = avg_rmse / len(matlist)
    print('The average rmse is:', avg_rmse)
    return avg_rmse


# PSNR
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
        elif channels == 32:
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




# SAM
def folder_img_SAM(generated_folder_path, groundtruth_folder_path):
    matlist = get_jpgs(generated_folder_path)
    avg_SAM = 0
    for i, matname in enumerate(matlist):
        generated_mat_path = os.path.join(generated_folder_path, matname)
        groundtruth_mat_path = os.path.join(groundtruth_folder_path, matname)
        generated_mat = hdf5.loadmat(generated_mat_path)['cube']
        groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube']
        SAM = computerSAM(generated_mat, groundtruth_mat)
        avg_SAM = avg_SAM + SAM
        # print('The %d-th mat\'s SAM:' % (i + 1), SAM)
    avg_SAM = avg_SAM / len(matlist)
    print('The average SAM is:', avg_SAM)
    return avg_SAM


# generated_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/Valid_Mosaic_pre_16"
# groundtruth_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/Valid_spectral_16"
#
#
# avg_mrae = folder_img_mrae(generated_folder_path, groundtruth_folder_path)
# avg_rmse = folder_img_rmse(generated_folder_path, groundtruth_folder_path)
# avg_PSNR = folder_img_PSNR(generated_folder_path, groundtruth_folder_path)
# avg_SAM = folder_img_SAM(generated_folder_path, groundtruth_folder_path)


