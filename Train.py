import os
import sys
import time
import datetime
import kornia
from torch import nn
from torch.utils.data import DataLoader
import torch
from MyDataset import MyDataset
from Net import DenseBlock as encoder1
from Net import BaseFeature as encoder2
from Net import DetailFeature as encoder3
from Net import BaseFeature as encoder4
from Net import DetailFeature as encoder5
from Net import Restormer_Decoder as decoder1
from Utils.Loss_function import ssim_loss
import Utils.Loss_function as loss_function
from Utils.Loss_function import Fusionloss
from Utils.Draw_loss_curve import Draw_loss_curve
criteria_fusion = Fusionloss()

lr = 1e-4
weight_decay = 0
optim_step = 1
optim_gamma = 0.5
epochs = 15
clip_grad_norm_value = 0.01
batch_size = 8
first_phase = 10
second_phase = 5
# Phase I
ir_local_ssim = 8
vi_global_ssim = 8
ir_mse = 2
vi_mse = 2
ir_grad = 20
vi_grad = 20
decomp_I = 2
# Phase II
ir_vi_ssim = 8
L1_parametric = 2
grad_parametric = 20
decomp_II = 2

def train():
    dataset = MyDataset(train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader = {'train': dataloader, }
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    prev_time = time.time()
    print("***********Dataloader Finished***********")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        Encoder1 = nn.DataParallel(encoder1()).to(device=device)
        Encoder2 = nn.DataParallel(encoder2()).to(device=device)
        Encoder3 = nn.DataParallel(encoder3()).to(device=device)
        Encoder4 = nn.DataParallel(encoder4()).to(device=device)
        Encoder5 = nn.DataParallel(encoder5()).to(device=device)
        Decoder = nn.DataParallel(decoder1()).to(device=device)
    elif torch.cuda.device_count() == 1:
        Encoder1 = encoder1().to(device=device)
        Encoder2 = encoder2().to(device=device)
        Encoder3 = encoder3().to(device=device)
        Encoder4 = encoder4().to(device=device)
        Encoder5 = encoder5().to(device=device)
        Decoder = decoder1().to(device=device)

    optimizer1 = torch.optim.Adam(
        Encoder1.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer2 = torch.optim.Adam(
        Encoder2.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer3 = torch.optim.Adam(
        Encoder3.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer4 = torch.optim.Adam(
        Encoder4.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer5 = torch.optim.Adam(
        Encoder5.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer6 = torch.optim.Adam(
        Decoder.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
    scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)
    scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=optim_step, gamma=optim_gamma)

    MSELoss = loss_function.MSELoss
    Local_SSIMLoss = loss_function.Local_SSIMLoss
    Global_SSIMLoss = loss_function.Global_SSIMLoss
    Gradient_loss = loss_function.Gradient_loss
    cc = loss_function.cc

    # model_pth = './model/NewFuse_28_11-04-00-02.pth'
    # Encoder1.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder1'])
    # Encoder2.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder2'])
    # Encoder3.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder3'])
    # Encoder4.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder4'])
    # Encoder5.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder5'])
    # Decoder.load_state_dict(torch.load(model_pth, weights_only=True)['Decoder'])

    mean_loss = []
    # epoch = 28
    for epoch in range(epochs):
    # while epoch <epochs:
        Temp_loss = 0
        for index, (ir_img, vi_img) in enumerate(dataloader):
            ir_img, vi_img = ir_img.to(device), vi_img.to(device)
            Encoder1.train()
            # Outlook_Attention 基础特征
            Encoder2.train()
            # INNmodules 细节特征
            Encoder3.train()
            Encoder4.train()
            Encoder5.train()
            Decoder.train()

            Encoder1.zero_grad()
            Encoder2.zero_grad()
            Encoder3.zero_grad()
            Encoder4.zero_grad()
            Encoder5.zero_grad()
            Decoder.zero_grad()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            optimizer5.zero_grad()
            optimizer6.zero_grad()
            if epoch<first_phase:
                share_feature = Encoder1(ir_img, vi_img)
                ir_B = Encoder2(ir_img)
                ir_D = Encoder3(ir_img)
                vi_B = Encoder4(vi_img)
                vi_D = Encoder5(vi_img)

                ir_ = Decoder(ir_img, share_feature, ir_B, ir_D, None, None)
                vi_ = Decoder(vi_img, share_feature, None, None, vi_B, vi_D)

                mse_loss_ir = ir_local_ssim  * Local_SSIMLoss(ir_img, ir_) + ir_mse * MSELoss(ir_img, ir_)
                mse_loss_vi = vi_global_ssim * Global_SSIMLoss(vi_img, vi_) + vi_mse * MSELoss(vi_img, vi_)
                cc_loss_B = cc(ir_B, vi_B)
                cc_loss_D = cc(ir_D, vi_D)
                # 关注
                loss_decomp = (cc_loss_D)**2 / (1.01+ cc_loss_B)
                ir_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(ir_img),
                                                        kornia.filters.SpatialGradient()(ir_))
                vi_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(vi_img),
                                                        kornia.filters.SpatialGradient()(vi_))
                loss = (mse_loss_ir + mse_loss_vi  + ir_grad * ir_fusion_gradient_loss +
                        vi_grad * vi_fusion_gradient_loss +decomp_I * loss_decomp )
                Temp_loss = Temp_loss + loss
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    Encoder1.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder2.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder3.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder4.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder5.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()
                optimizer5.step()
                optimizer6.step()
            else:
                share_feature = Encoder1(ir_img, vi_img)
                ir_B = Encoder2(ir_img)
                ir_D = Encoder3(ir_img)
                vi_B = Encoder4(vi_img)
                vi_D = Encoder5(vi_img)

                fusion_feature = Decoder(vi_img, share_feature, ir_B, vi_B, ir_D, vi_D)

                mse_loss_ir = ir_local_ssim * Local_SSIMLoss(ir_img, fusion_feature) + ir_mse * MSELoss(ir_img, fusion_feature)
                mse_loss_vi = vi_global_ssim * Global_SSIMLoss(vi_img,fusion_feature) + vi_mse * MSELoss(vi_img, fusion_feature)
                cc_loss_B = cc(ir_B, vi_B)
                cc_loss_D = cc(ir_D, vi_D)
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
                ir_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(ir_img),
                                                        kornia.filters.SpatialGradient()(fusion_feature))
                vi_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(vi_img),
                                                        kornia.filters.SpatialGradient()(fusion_feature))
                # loss = (mse_loss_ir + mse_loss_vi + ir_grad * ir_fusion_gradient_loss +
                #         vi_grad * vi_fusion_gradient_loss +decomp*loss_decomp)
                loss_,_,_ = criteria_fusion(ir_img,vi_img,fusion_feature,L1_parametric,grad_parametric)
                ir_vi_SSIM = ssim_loss(ir_img,vi_img,fusion_feature)
                loss = loss_ + decomp_II * loss_decomp + ir_vi_ssim * ir_vi_SSIM
                Temp_loss = Temp_loss + loss
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    Encoder1.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder2.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder3.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder4.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder5.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()
                optimizer5.step()
                optimizer6.step()

            batches_done = epoch * len(loader['train']) + index
            batches_left = epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
                % (
                    epoch+1,
                    epochs,
                    index + 1,
                    len(loader['train']),
                    loss.item(),
                    time_left,
                )
            )
            sys.stdout.flush()

        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()
        scheduler5.step()
        scheduler6.step()

        if optimizer1.param_groups[0]['lr'] <= 1e-6:
            optimizer1.param_groups[0]['lr'] = 1e-6
        if optimizer2.param_groups[0]['lr'] <= 1e-6:
            optimizer2.param_groups[0]['lr'] = 1e-6
        if optimizer3.param_groups[0]['lr'] <= 1e-6:
            optimizer3.param_groups[0]['lr'] = 1e-6
        if optimizer4.param_groups[0]['lr'] <= 1e-6:
            optimizer4.param_groups[0]['lr'] = 1e-6
        if optimizer5.param_groups[0]['lr'] <= 1e-6:
            optimizer5.param_groups[0]['lr'] = 1e-6
        if optimizer6.param_groups[0]['lr'] <= 1e-6:
            optimizer6.param_groups[0]['lr'] = 1e-6

        mean_loss.append(Temp_loss/(dataset.__len__()/batch_size))
        if True:
            checkpoint = {
                'Encoder1': Encoder1.state_dict(),
                'Encoder2': Encoder2.state_dict(),
                'Encoder3': Encoder3.state_dict(),
                'Encoder4': Encoder4.state_dict(),
                'Encoder5': Encoder5.state_dict(),
                'Decoder': Decoder.state_dict(),
            }
            if not os.path.isdir('./model'):
                os.mkdir('./model')
            torch.save(checkpoint, os.path.join(f"./model/NewFuse_{epoch+1}_" + timestamp + '.pth'))
        # epoch = epoch + 1
    Draw_loss_curve(epochs, Mean_Loss=torch.tensor(mean_loss).cpu(),run_time=timestamp)
if __name__ == '__main__':
    train()