import os
import numpy as np
from torch import nn
import torch
from Net import DenseBlock as encoder1
from Net import BaseFeature as encoder2
from Net import DetailFeature as encoder3
from Net import BaseFeature as encoder4
from Net import DetailFeature as encoder5
from Net import Restormer_Decoder as decoder1
from Utils.Image_read_and_save import img_save, image_read_cv2
from Utils.Valuation import Valuation

model_pth = './model/NewFuse_15_11-05-00-27.pth'
def val():
    for dataset_name in ['TNO','RoadScene','MSRS','MRI_CT']:
        Model_Name = 'MBSFuse'
        print("The test result of " + dataset_name + ' :')
        test_folder = os.path.join('test_images', dataset_name)
        test_out_folder = os.path.join('Results', dataset_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.device_count() > 1:
            Encoder1 = nn.DataParallel(encoder1()).to(device=device)
            Encoder2 = nn.DataParallel(encoder2()).to(device=device)
            Encoder3 = nn.DataParallel(encoder3()).to(device=device)
            Encoder4 = nn.DataParallel(encoder4()).to(device=device)
            Encoder5 = nn.DataParallel(encoder5()).to(device=device)
            Decoder = nn.DataParallel(decoder1()).to(device=device)
        else:
            Encoder1 = encoder1().to(device)
            Encoder2 = encoder2().to(device)
            Encoder3 = encoder3().to(device)
            Encoder4 = encoder4().to(device)
            Encoder5 = encoder5().to(device)
            Decoder = decoder1().to(device)

        Encoder1.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder1'])
        Encoder2.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder2'])
        Encoder3.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder3'])
        Encoder4.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder4'])
        Encoder5.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder5'])
        Decoder.load_state_dict(torch.load(model_pth, weights_only=True)['Decoder'])
        Encoder1.eval()
        Encoder2.eval()
        Encoder3.eval()
        Encoder4.eval()
        Encoder5.eval()
        Decoder.eval()
        with torch.no_grad():
            for img_name in os.listdir(os.path.join(test_folder, 'ir')):
                data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[None, None, ...]/255.0
                data_VIS = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='GRAY')[None, None, ...] / 255.0
                ir_img, vi_img = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
                ir_img, vi_img = ir_img.to(device), vi_img.to(device)
                share_feature = Encoder1(ir_img, vi_img)
                ir_detail = Encoder2(ir_img)
                ir_base = Encoder3(ir_img)
                vi_detail = Encoder4(vi_img)
                vi_base = Encoder5(vi_img)
                fusion_feature = Decoder(vi_img, share_feature, ir_detail, ir_base, vi_detail, vi_base)

                data_normalized = (fusion_feature - torch.min(fusion_feature)) / (torch.max(fusion_feature) - torch.min(fusion_feature))
                data_scaled = (data_normalized * 255).cpu().numpy()
                fi = np.squeeze(data_scaled).astype(np.uint8)
                img_save(fi.astype(np.uint8), img_name.split(sep='.')[0], test_out_folder)

        eval_folder = test_out_folder
        ori_img_folder = test_folder
        metric_result = np.zeros((9))
        for img_name in os.listdir(os.path.join(ori_img_folder, "ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0] + ".png"), 'GRAY')
            metric_result += np.array([Valuation.EN(fi), Valuation.SD(fi)
                                          , Valuation.SF(fi), Valuation.MI(fi, ir, vi)
                                          , Valuation.SCD(fi, ir, vi), Valuation.VIFF(fi, ir, vi),
                                            Valuation.Qabf(fi,ir,vi), Valuation.MSE(fi, ir, vi),
                                       Valuation.SSIM(fi, ir, vi)])

        metric_result /= len(os.listdir(eval_folder))
        print("\t\t\t EN\t\t  SD\t SF\t     MI\t     SCD\t VIF\t Qabf\t MSE\t SSIM")
        print(f"{Model_Name}" + '\t' + str(np.round(metric_result[0], 2)) + '\t'
              + str(np.round(metric_result[1], 2)) + '\t'
              + str(np.round(metric_result[2], 2)) + '\t'
              + str(np.round(metric_result[3], 2)) + '\t'
              + str(np.round(metric_result[4], 2)) + '\t'
              + str(np.round(metric_result[5], 2)) + '\t'
              + str(np.round(metric_result[6], 2)) + '\t'
              + str(np.round(metric_result[7], 2)) + '\t'
              + str(np.round(metric_result[8], 2))
              )

if __name__ == '__main__':
    val()