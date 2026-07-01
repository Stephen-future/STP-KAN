import argparse
import os
import cv2
import numpy as np
import hdf5storage as hdf5
from official_scoring_code2.EvalMetrics import computeMRAE,computeRMSE,computerPSNR ,computerSAM
from skimage.metrics import structural_similarity as ssim


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
        generated_mat = hdf5.loadmat(generated_mat_path)['cube'] # shape: (482, 512, 16)
        groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube'] # shape: (482, 512, 16)
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
    for i, matname in enumerate(matlist):
        generated_mat_path = os.path.join(generated_folder_path, matname)
        groundtruth_mat_path = os.path.join(groundtruth_folder_path, matname)
        generated_mat = hdf5.loadmat(generated_mat_path)['cube']
        groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube']
        PSNR = computerPSNR(generated_mat, groundtruth_mat)
        avg_PSNR = avg_PSNR + PSNR
        # print('The %d-th mat\'s PSNR:' % (i + 1), PSNR)
    avg_PSNR = avg_PSNR / len(matlist)
    print('The average PSNR is:', avg_PSNR)
    return avg_PSNR



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

# SSIM
def folder_img_SSIM(generated_folder_path, groundtruth_folder_path):
    """
    Calculate the average SSIM between hyperspectral images in two folders.

    Parameters:
    - generated_folder_path: Path to the folder containing generated images.
    - groundtruth_folder_path: Path to the folder containing ground truth images.

    Returns:
    - avg_SSIM: The average SSIM value across all image pairs.
    """
    matlist = get_jpgs(generated_folder_path)  # List of image names in the folder
    avg_SSIM = 0  # Initialize average SSIM accumulator

    for i, matname in enumerate(matlist):
        # Construct full paths for the generated and ground truth image files
        generated_mat_path = os.path.join(generated_folder_path, matname)
        groundtruth_mat_path = os.path.join(groundtruth_folder_path, matname)

        # Load the .mat files and extract the 'cube' key
        generated_mat = hdf5.loadmat(generated_mat_path)['cube']
        groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube']

        # Compute SSIM across corresponding spectral bands
        total_ssim = 0
        band_count = generated_mat.shape[2]  # Assuming the 3rd dimension represents spectral bands

        for band in range(band_count):
            generated_band = generated_mat[:, :, band]
            groundtruth_band = groundtruth_mat[:, :, band]
            band_ssim = ssim(generated_band, groundtruth_band,
                             data_range=groundtruth_band.max() - groundtruth_band.min())
            total_ssim += band_ssim

        # Average SSIM for current hyperspectral cube
        cube_ssim = total_ssim / band_count
        avg_SSIM += cube_ssim  # Add to overall average

        # Uncomment below to see individual SSIM values
        # print(f'The {i + 1}-th mat\'s SSIM: {cube_ssim}')

    # Compute final average SSIM
    avg_SSIM /= len(matlist)
    print('The average SSIM is:', avg_SSIM)
    return avg_SSIM

# MSAM
def compute_spectral_angle(generated, groundtruth):
    """
    Compute the spectral angle for a single pixel's spectra.

    Parameters:
    - generated: Reconstructed spectrum as a 1D array.
    - groundtruth: Ground truth spectrum as a 1D array.

    Returns:
    - Spectral angle in radians.
    """
    numerator = np.dot(generated, groundtruth)
    denominator = np.linalg.norm(generated) * np.linalg.norm(groundtruth)
    if denominator == 0:  # To handle cases where the norm is zero
        return 0
    return np.arccos(np.clip(numerator / denominator, -1, 1))


def folder_img_MSAM(generated_folder_path, groundtruth_folder_path):
    """
    Calculate the average MSAM between hyperspectral images in two folders.

    Parameters:
    - generated_folder_path: Path to the folder containing generated images.
    - groundtruth_folder_path: Path to the folder containing ground truth images.

    Returns:
    - avg_MSAM: The average MSAM value across all image pairs.
    """
    matlist = get_jpgs(generated_folder_path)  # List of image names in the folder
    avg_MSAM = 0  # Initialize average MSAM accumulator

    for i, matname in enumerate(matlist):
        # Construct full paths for the generated and ground truth image files
        generated_mat_path = os.path.join(generated_folder_path, matname)
        groundtruth_mat_path = os.path.join(groundtruth_folder_path, matname)

        # Load the .mat files and extract the 'cube' key
        generated_mat = hdf5.loadmat(generated_mat_path)['cube']
        groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube']

        # Verify the dimensions match
        if generated_mat.shape != groundtruth_mat.shape:
            raise ValueError(f"Shape mismatch between generated and ground truth for {matname}")

        # Compute MSAM for all pixels
        height, width, bands = generated_mat.shape
        total_sam = 0
        pixel_count = height * width

        for x in range(height):
            for y in range(width):
                generated_spectrum = generated_mat[x, y, :]
                groundtruth_spectrum = groundtruth_mat[x, y, :]
                total_sam += compute_spectral_angle(generated_spectrum, groundtruth_spectrum)

        # Average MSAM for the current hyperspectral image
        cube_msam = total_sam / pixel_count
        avg_MSAM += cube_msam

        # Uncomment below to see individual MSAM values
        # print(f'The {i + 1}-th mat\'s MSAM: {cube_msam}')

    # Compute final average MSAM
    avg_MSAM /= len(matlist)
    print('The average MSAM is:', avg_MSAM)
    return avg_MSAM


def compute_mse(img1, img2):
    """计算均方误差 (MSE)"""
    return np.mean((img1 - img2) ** 2)


def folder_img_ERGAS(generated_folder_path, groundtruth_folder_path, scale_ratio=1):
    """
    计算两个文件夹中的高光谱图像的平均 ERGAS 指标。

    参数：
    - generated_folder_path: 生成图像的文件夹路径
    - groundtruth_folder_path: 真实图像的文件夹路径
    - scale_ratio: 分辨率缩放因子 (默认为4)

    返回：
    - avg_ERGAS: 平均 ERGAS 值
    """
    matlist = get_jpgs(generated_folder_path)  # 获取图像文件名列表
    total_ergas = 0

    for i, matname in enumerate(matlist):
        # 生成图像路径 & 真实图像路径
        generated_mat_path = os.path.join(generated_folder_path, matname)
        groundtruth_mat_path = os.path.join(groundtruth_folder_path, matname)

        # 加载 .mat 文件，并提取 'cube' 数据
        generated_mat = hdf5.loadmat(generated_mat_path)['cube']
        groundtruth_mat = hdf5.loadmat(groundtruth_mat_path)['cube']

        # 确保两张图像的尺寸匹配
        assert generated_mat.shape == groundtruth_mat.shape, "Image dimensions do not match!"

        band_count = generated_mat.shape[2]  # 获取光谱波段数量
        mse_list = []
        mu_list = []

        for band in range(band_count):
            generated_band = generated_mat[:, :, band]
            groundtruth_band = groundtruth_mat[:, :, band]

            # 计算 MSE 和 真实图像均值
            mse_list.append(compute_mse(generated_band, groundtruth_band))
            mu_list.append(np.mean(groundtruth_band))

        # 计算 ERGAS
        mse_array = np.array(mse_list)
        mu_array = np.array(mu_list)
        ergas_value = 100 * np.sqrt(np.mean(mse_array / (mu_array ** 2))) / scale_ratio

        total_ergas += ergas_value  # 累加每个图像的 ERGAS

    avg_ERGAS = total_ergas / len(matlist)  # 计算平均 ERGAS
    print('The average ERGAS is:', avg_ERGAS)
    return avg_ERGAS

# generated_folder_path = "D:/code_UNet3/Result_valid/unet/0"  #unet
# generated_folder_path = "D:/code_UNet3/Result_valid/unet_MAB/0"  #CFWB+Unet+MAB
# generated_folder_path = "D:/code_UNet3/logs_CFWB_unet/exp/valid_rec_results"   #CFWB+unet
# generated_folder_path = "D:/code_UNet3/logs_res2unet_se/exp/valid_rec_results"  #Res2unet
# generated_folder_path = "D:/code_UNet3/logs-mcan/exp/valid_rec_results"  #MCAN
# generated_folder_path = "D:/code_UNet3/Result_valid/man/0"  #man
# generated_folder_path = "D:/code_UNet3/Result/Dence_unet/0"  #dence_unet
# generated_folder_path = "D:/code_UNet3/Result/unet_MLKA/0"  #unet_MLKA


# generated_folder_path = "D:/code_UNet3/Result/CFWB+unet/0"
# generated_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/Valid_Mosaic_pre"  #33.894408647852146
# generated_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/Valid_Mosaic_16"   #15.266351130747314


groundtruth_folder_path = "/home/ubuntu/Downloads/wz/MSFA/data/Valid_spectral_crop_16_2"
# groundtruth_folder_path = "/home/ubuntu/Downloads/wz/MSFA/data/CAVE/Valid_spectral_16"
generated_folder_path = "/home/ubuntu/Downloads/wz/MSFA/data/result/ablation/ResBlock*5/exp/valid_rec_results_2"   # /exp/valid_rec_results
# generated_folder_path = "/home/ubuntu/Downloads/wz/MSFA/data/result/RF_Unet/8_4*4/exp/valid_rec_results"
# generated_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/Train_mosaic_16"
# groundtruth_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/Train_spectral_16"


#传统方法
# generated_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/ct/BTES"   #I_BTES1
# generated_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/ct/GRMR"   #I_GRMR_rec1
# generated_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/ct/PPID"   #I_PPID1
# generated_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/ct/WB"     #I_WB1
# groundtruth_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/Valid_spectral_16"
# generated_folder_path = "D:/NTIRE2022/Mosaic-HS dataset/ct/direc"   #mosaic 方向插值

# generated_folder_path = "D:/code_UNet3/chapter4/RF_Unet/0"
# generated_folder_path = "D:/code_UNet3/chapter4/unet+res2fft/0"
# generated_folder_path = "D:/code_UNet3/chapter4/unet+res2fft+attention/0"
# generated_folder_path = "D:/code_UNet3/chapter4/unet+res2net/0"


avg_mrae = folder_img_mrae(generated_folder_path, groundtruth_folder_path)
avg_rmse = folder_img_rmse(generated_folder_path, groundtruth_folder_path)
avg_PSNR = folder_img_PSNR(generated_folder_path, groundtruth_folder_path)
avg_SSIM = folder_img_SSIM(generated_folder_path, groundtruth_folder_path)
# avg_SAM = folder_img_SAM(generated_folder_path, groundtruth_folder_path)
avg_MSAM = folder_img_MSAM(generated_folder_path, groundtruth_folder_path)
avg_ERGAS = folder_img_ERGAS(generated_folder_path, groundtruth_folder_path)

print('MRAE: {},RMSE: {},PSNR: {},SSIM: {},MSAM: {},ERGAS: {}'.format(avg_mrae, avg_rmse, avg_PSNR, avg_SSIM, avg_MSAM, avg_ERGAS))
