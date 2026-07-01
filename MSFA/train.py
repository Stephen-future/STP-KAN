import argparse
import os
import trainer2_same    # STP-KAN


# from trainer import Trainer,save_checkpoint

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Saving, and loading parameters

    # 开始训练的epoch  默认为0 如果resume 要改这里 和rusume 和latest path
    parser.add_argument('--start_epoch', default=1000)
    parser.add_argument('--val_epoch', default=10, help='interval between model valid(by epochs)')

    parser.add_argument('--resume', default=True,help = "loading networks model parameters:True or Fasle")
    parser.add_argument('--latest_path', default=r"./rgb/RGB_SSMamba2/best_1000epoch.pth", help="the file of networks model parameters")
    parser.add_argument('--freeze_layers', default=True, help="loading networks model parameters:True or Fasle")

    parser.add_argument('--test_batch_size', default=1)
    parser.add_argument('--save_path', type = str, default = 'track2', choices = ['track1', 'track2'], help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')    # default = 500
    parser.add_argument('--save_by_iter', type = int, default = 10, help = 'interval between model checkpoints (by iterations)')   # default = 100000
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = True, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    # parser.add_argument("--outf", type=str, default="Results Mosaic", help='path log files')    # path log files

    # Training parameters
    parser.add_argument('--epochs', type = int, default = 1001, help = 'number of epochs of training that ensures 100K training iterations')   # default = 12001
    parser.add_argument('--batch_size', type = int, default = 4, help = 'size of the batches, 8 is recommended')       # 4
    parser.add_argument('--seed', type=int, default=1, help='Random_seed')
    parser.add_argument('--lr', type = float, default = 0.0001, help = 'Adam: learning rate')  #0.0001                   #   Ranger  Adam
    parser.add_argument('--b1', type = float, default = 0.9, help = 'Adam: decay of first order momentum of gradient')   # RF_unet 0.9
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient') # RF_unet 0.999
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')                # 0
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode, by_epoch or by_iter')      #
    parser.add_argument('--lr_decrease_epoch', type = int, default = 200, help = 'lr decrease at certain epoch and its multiple')   # default = 2500   RF_unet 200
    parser.add_argument('--lr_decrease_iter', type = int, default = 50000, help = 'lr decrease at certain epoch and its multiple')   # default = 50000
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 3, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_half', type = float, default = 0.5, help = 'lambda_half for SubNet')

   # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'relu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--dim', type=int, default=128, help = 'transformer')
    parser.add_argument("--num_channel", type=int, default=16)
    parser.add_argument('--in_channels', type = int, default = 16, help = 'input channels for generator')    # 16
    parser.add_argument('--out_channels', type = int, default = 16, help = 'output channels for generator')  # 16
    parser.add_argument('--start_channels', type = int, default = 32, help = 'start channels for generator') #
    parser.add_argument('--n_feat',type = int,default = 16,help = 'input channels for generator')
    parser.add_argument('--stage',type = int,default= 3,help = 'level')
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')   # xavier
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')

# Dataset parameters
    parser.add_argument('--augment', type=bool, default=True, help='Data augment')
    parser.add_argument('--baseroot', type = str, default = './', help = 'baseroot')
    parser.add_argument('--data_mask', default=r'./data/mask_16', help="train dataset input")
    parser.add_argument('--train_data_in', default=r'./data/Train_CFWB_16', help="train dataset input")
    parser.add_argument('--train_data_reference', default=r'./data/Train_spectral_crop_16', help="train dataset reference")
    parser.add_argument('--train_data_in_2', default=r'./data/Train_CFWB_16_2', help="train dataset input")
    parser.add_argument('--train_data_reference_2', default=r'./data/Train_spectral_crop_16_2', help="train dataset reference")
    # parser.add_argument('--train_data_reference_3', default=r'./data/Train_spectral_crop_25',help="train dataset reference")
    parser.add_argument('--train_data_predict', default='D:/NTIRE2022/code_Mosaic/val_out', help="train dataset predict output")
    parser.add_argument('--val_data_in', default=r'./data/Valid_CFWB_16', help="validation dataset input")
    parser.add_argument('--val_data_in_2', default=r'./data/Valid_CFWB_16_2', help="validation dataset input")
    # parser.add_argument('--val_data_in_3', default=r'./data/Valid_spectral_crop_25', help="validation dataset input")
    parser.add_argument('--val_save_path', default='./data/log/exp/valid_rec_results', help="validation dataset predict output")
    parser.add_argument('--val_save_path_2', default='./data/log/exp/valid_rec_results_2', help="validation dataset predict output")
    # parser.add_argument('--val_save_path_3', default='./data/log/exp/valid_rec_results_3',help="validation dataset predict output")
    parser.add_argument('--crop_size', type = int, default = 240, help = 'crop size')     # crop size，64//128// 256// 320// 384

    parser.add_argument('--log_dir', default=r'./data/log', help='path log files')
    parser.add_argument('--name', default='exp', help="实验记录log描述")

    opt = parser.parse_args()

    # ----------------------------------------
    #                 Trainer
    # ----------------------------------------
    trainer2_same.Trainer(opt)