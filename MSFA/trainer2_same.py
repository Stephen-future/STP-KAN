import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler
import torch.backends.cudnn as cudnn
import h5py
import hdf5storage as hdf5
import dataset2
import utils
from utils import initialize_logger,record_loss
from compute import folder_img_PSNR
import losses

def save_matv73(mat_name, var_name, var):
    hdf5.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)
def saveCube(path, cube, bands=None):
    hdf5.write({u'cube': cube,u'bands': bands,    }, '.',path, matlab_compatible=True)


def Trainer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs:" % (gpu_num))
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    opt.log_path = os.path.join(opt.log_dir, opt.name)     #### 记录log的路径都用 opt.log_path
    opt.val_save_path = os.path.join(opt.log_path, 'valid_rec_results')
    os.makedirs(opt.val_save_path, exist_ok=True)
    opt.val_save_path_2 = os.path.join(opt.log_path, 'valid_rec_results_2')
    os.makedirs(opt.val_save_path_2, exist_ok=True)
    # Create folders
    save_model_folder = opt.log_path
    os.makedirs(save_model_folder, exist_ok=True)
    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # ###################=============================================
    # Loss functions
    # pytorch损失函数 —— https://blog.csdn.net/java_hzp/article/details/103357966
    criterion_L1 = torch.nn.L1Loss().cuda()# L1 和 MAE 一样
    loss_FFT = losses.fftLoss()


    # Initialize SGN , utils.py 中create_generator(opt)   ，  generator = network_code1.SGN(opt)
    generator = utils.create_generator(opt)
    print('Parameters number is ', sum(param.numel() for param in generator.parameters()))

    if opt.resume is True:
        trained_dict = torch.load(opt.latest_path, map_location="cpu")
        utils.load_dict(generator, trained_dict)
        print("load {} model success!".format(opt.latest_path))

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    # 优化器就是需要根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    # eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
    # visualzation    我加的
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    loss_csv = open(os.path.join(opt.log_dir, 'loss.csv'), 'a+')        # 产生了一个文件 loss.csv，保存在"CleanResults"
    # loss_txt = open(os.path.join(opt.log_path, 'loss.txt'), 'a+')        # 产生了一个文件 loss.txt，保存在"CleanResults"
    log_dir1 = os.path.join(opt.log_dir, 'train.log')                    # 产生了一个文件 train.log
    logger = initialize_logger(log_dir1)        # utils.py

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))             # lr_decrease_factor = 0.5
            print("lr: %s " % (lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # Save the model
    # 主要区别在于 generator.module.后面是否有【 .state_dict() 】
    # torch.save(generator.module.state_dict(), save_model_path_params)   #  .state_dict() , 只保存了网络参数， (速度快, 占内存少)，用netron软件打不开
    # torch.save(generator.module, save_model_path)
    def save_model(opt, epoch, iteration, len_dataset, generator):
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            # model_name = 'G_epoch%d_bs%d.pth' % (epoch, opt.batch_size)        # 我加的，保存完整结构，不可以调用的模型
            model_name_params = 'G_epoch%d_bs%d_params.pth' % (epoch, opt.batch_size)     # 原代码，但是我加了_params，只保存了网络参数，可以调用的模型
        if opt.save_mode == 'iter':
            # model_name = 'G_iter%d_bs%d.pth' % (iteration, opt.batch_size)
            model_name_params = 'G_iter%d_bs%d_params.pth' % (iteration, opt.batch_size)
        # save_model_path = os.path.join(opt.save_path, model_name)
        save_model_path_params = os.path.join(opt.save_path, model_name_params)
        # Save model
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path_params)   #  .state_dict() , 只保存了网络参数， (速度快, 占内存少)，用netron软件打不开
                    # torch.save(generator.module, save_model_path)         # 我改的，保存完整结构【generator = model】,但是保存的模型无法调用，能用netron软件打开
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path_params)
                    # torch.save(generator.module, save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path_params)
                    # torch.save(generator, save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path_params)
                    # torch.save(generator, save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    def save_model_by(opt, best=False, epoch=0):
        if best is True:
            save_model_path_params = os.path.join(opt.log_path, "best_{}epoch.pth".format(epoch))
        else:
            save_model_path_params = os.path.join(opt.log_path, "latest.pth")

        if opt.multi_gpu == True:
           torch.save(generator.module.state_dict(), save_model_path_params)  # .state_dict() , 只保存了网络参数， (速度快, 占内存少)，用netron软件打不开
           # torch.save(generator.module, save_model_path)         # 保存完整结构【generator = model】,但是保存的模型无法调用，能用netron软件打开

        else:
            torch.save(generator.module.state_dict(), save_model_path_params)
            # torch.save(generator, save_model_path)



    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------
    # Define the dataset
    trainset = dataset2.HS_multiscale_DSet(opt)                     # HS_multiscale_DSet 是 dataset.py里的
    print('The overall number of train images:', len(trainset))        # 训练集数据的数量
    # Define the dataloader  数据加载器，from torch.utils.data import DataLoader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)

    # 验证集数据读取
    test_dataset = dataset2.HS_multiscale_ValDSet(opt)
    print('The overall number of test images:', len(test_dataset))        # 验证集数据的数量
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False,num_workers=opt.num_workers, pin_memory=True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------
    # Count start time
    start_time = time.time()

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # For loop training
    best_PSNR = 35
    best_epoch = 0
    # 初始化损失列表
    loss_list_9 = []
    loss_list_16 = []
    loss_list_25 = []
    for epoch in range(opt.start_epoch, opt.epochs):
        epoch_loss_9 = 0
        epoch_loss_16 = 0
        epoch_loss_25 = 0
        for i, (mosaic_16, img_A, mosaic_9, img_C) in enumerate(dataloader):
            # To device
            mosaic_16 = mosaic_16.cuda()  # 马赛克数据
            img_A = img_A.cuda()  # 原HS图像B
            b, c, h, w = img_A.shape

            mosaic_9 = mosaic_9.cuda()  # 马赛克数据
            img_C = img_C.cuda()  # 原HS图像B
            b2, c2, h2, w2 = img_C.shape
            generator.train()
            # img_A = np.array(img_A).transpose(2, 0, 1)    #  (16, 480, 512)
            # Train Generator
            # optimizer_G.zero_grad()
            recon_A, recon_C = generator(mosaic_16, c, mosaic_9, c2)  # 原RGB图像A   经过网络后   生成HS图像B
            # recon_A = generator(mosaic_16)


            # 根据输入数据设置不同的损失权重
            if c == 9:
                loss1 = criterion_L1(recon_A, img_A) * 1  # 0.52
                epoch_loss_9 += loss1.item()  # 累积9通道的损失
            elif c == 16:
                loss1 = criterion_L1(recon_A, img_A) * 1  # 0.31
                epoch_loss_16 += loss1.item()  # 累积16通道的损失
            else:
                loss1 = criterion_L1(recon_A, img_A) * 1  # 0.17
                epoch_loss_25 += loss1.item()  # 累积25通道的损失
            if c2 == 9:
                loss2 = criterion_L1(recon_C, img_C) * 1  # 0.52
                epoch_loss_9 += loss2.item()  # 累积9通道的损失
            elif c2 == 16:
                loss2 = criterion_L1(recon_C, img_C) * 1  # 0.31
                epoch_loss_9 += loss2.item()  # 累积16通道的损失
            else:
                loss2 = criterion_L1(recon_C, img_C) * 1  # 0.17
                epoch_loss_25 += loss2.item()  # 累积25通道的损失

            loss = loss1 + loss2
            loss.backward()  # 向后损失，通过loss.backward()，来对梯度进行反向传播。计算参数更新值

            # 每 3 个批次更新一次梯度
            if (i + 1) % 1 == 0 or (i + 1) == len(dataloader):
                optimizer_G.step()  # 更新模型参数
                optimizer_G.zero_grad()  # 清空上一步的残余更新参数值

            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - start_time))
            start_time = time.time()

            # ====================================================
            # print loss
            print("Epoch [%d/%d], [Iters %d/%d], Train Loss: %.9f, Time_left: %s "
                  % ((epoch + 1), opt.epochs, i, len(dataloader), loss.item(), time_left))
            # save loss        ， Print log
            record_loss(loss_csv, epoch, i, loss.item())  # 产生log文件
            # record_loss(loss_txt, epoch, i, loss.item())          # 产生log文件
            logger.info("Epoch [%d/%d], [Iters %d/%d], Train Loss: %.9f, Time_left: %s "
                        % ((epoch + 1), opt.epochs, i, len(dataloader), loss.item(), time_left))

            # ====================================================
            save_model_by(opt, False)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)

        # 计算每个 epoch 的平均损失，并保存到对应的列表中
        if epoch_loss_9 > 0:
            avg_loss_9 = epoch_loss_9 / len(dataloader)
            loss_list_9.append(avg_loss_9)

        if epoch_loss_16 > 0:
            avg_loss_16 = epoch_loss_16 / len(dataloader)
            loss_list_16.append(avg_loss_16)

        if epoch_loss_25 > 0:
            avg_loss_25 = epoch_loss_25 / len(dataloader)
            loss_list_25.append(avg_loss_25)

        # 每100个epoch绘制一次损失图
        if (epoch + 1) % 100 == 0:
            plt.figure(figsize=(10, 6))

            # 绘制9通道损失
            if loss_list_9:
                plt.plot(range(opt.start_epoch, epoch + 1), loss_list_9, label='9 Channels Loss', color='r')

            # 绘制16通道损失
            if loss_list_16:
                plt.plot(range(opt.start_epoch, epoch + 1), loss_list_16, label='16 Channels Loss', color='g')

            # 绘制25通道损失
            if loss_list_25:
                plt.plot(range(opt.start_epoch, epoch + 1), loss_list_25, label='25 Channels Loss', color='b')

            # 设置图例和标题
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss for Different Channels (Epochs {opt.start_epoch} to {epoch + 1})')
            plt.legend()
            plt.grid(True)

            # 保存损失图
            plt.savefig(f'./data/log/loss_epoch_{epoch + 1}.png')
            # 关闭当前绘图，释放内存
            plt.close()
        #### 验证  opt.val_epoch=10
        if epoch % opt.val_epoch == 0:
            for j, (img, path, img2, path2) in enumerate(test_loader):
                # To device
                img1 = img.cuda()
                b, c, h, w = img1.shape
                path = path[0]

                img2 = img2.cuda()
                b2, c2, h2, w2 = img2.shape
                path2 = path2[0]
                generator.eval()
                # Forward propagation
                with torch.no_grad():
                    out, out2 = generator(img1, c, img2, c2)
                cube = out.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
                cube2 = out2.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
        ### AVIRIS ###
                if c == 9:
                    save_img_name = path[:12] +'_9'+ '.mat'
                elif c == 16:
                    save_img_name = path[:12] +'_16'+ '.mat'      ### ARAD:12; AVIRIS:11
                elif c == 25:
                    save_img_name = path[:12] +'_25'+ '.mat'
                else:
                    raise ValueError("Unsupported input channel size: {}".format(c))
                if c2 == 9:
                    save_img_name2 = path2[:12] +'_9'+ '.mat'
                elif c2 == 16:
                    save_img_name2 = path2[:12] +'_16'+ '.mat'    ### ARAD:12; AVIRIS:11
                elif c2 == 25:
                    save_img_name2 = path2[:12] +'_25'+ '.mat'
                else:
                    raise ValueError("Unsupported input channel size: {}".format(c))

                # save_img_name = path[:12]  + '.mat'
                save_img_path = os.path.join(opt.val_save_path, save_img_name)
                save_img_path_2 = os.path.join(opt.val_save_path_2, save_img_name2)
                # save_img_path = os.path.join(opt.val_data_predict, save_img_name)
                # saveCube(os.path.join(save_img_path), cube, bands=None)
                save_matv73(save_img_path, 'cube', cube)
                save_matv73(save_img_path_2, 'cube', cube2)

        # 真值高光谱位置
        #     val_Spec_path = r'./data/AVIRIS/Valid_spectral_16'
        #     val_Spec_path_2 = r'./data/AVIRIS/Valid_spectral_16_2'
            val_Spec_path = r'./data/Valid_spectral_crop_16'
            val_Spec_path_2 = r'./data/Valid_spectral_crop_16_2'
            result = folder_img_PSNR(opt.val_save_path, val_Spec_path)
            result2 = folder_img_PSNR(opt.val_save_path_2, val_Spec_path_2)

            avg_psnr_3ch = result['avg_psnr_3ch'] + result2['avg_psnr_4ch']
            avg_psnr_4ch = result['avg_psnr_4ch'] + result2['avg_psnr_3ch']
            avg_psnr_5ch = result['avg_psnr_5ch'] + result2['avg_psnr_5ch']
            total_avg_psnr = result['total_avg_psnr']
            # total_avg_psnr = result2['total_avg_psnr']

            if total_avg_psnr > best_PSNR:
                save_model_by(opt, True, epoch)
                best_PSNR = total_avg_psnr
                best_epoch = epoch

            print("valid========>epoch:{}, avg_psnr_3ch:{}, avg_psnr_4ch:{}, avg_psnr_5ch:{}, avg_PSNR:{}, best_epoch:{}, best_PSNR:{}".format(epoch, avg_psnr_3ch, avg_psnr_4ch, avg_psnr_5ch, total_avg_psnr, best_epoch, best_PSNR))
            with open(os.path.join(opt.log_path, 'valid_log.txt'), 'a+') as f:
                print("valid========>epoch:{}, avg_psnr_3ch:{}, avg_psnr_4ch:{}, avg_psnr_5ch:{}, avg_PSNR:{}, best_epoch:{}, best_PSNR:{}".format(epoch, avg_psnr_3ch, avg_psnr_4ch, avg_psnr_5ch, total_avg_psnr, best_epoch, best_PSNR), file=f)


