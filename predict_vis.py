import argparse
import os
import pprint
import yaml
import torch
from torch import nn

from dataset.semi import SemiDataset
from utils.utils import *
from model.SEUNet import se_resnext50_32x4d
# from model.SEUNet_CPM import se_resnext50_32x4d # for MCSS
# from model.SEUNet_base import se_resnext50_32x4d # for others

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

from torch import optim
from utils.custom_eval_model import set_to_eval
from torch.utils.data import DataLoader
from tqdm import tqdm
# import torch.multiprocessing
# from torch.cuda.amp import autocast
# from torch.cuda.amp import grad_scaler

import csv
import numpy as np
from PIL import Image

# torch.multiprocessing.set_sharing_strategy('file_system') # 使用文件系统作为数据共享的策略。
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

DATASET = 'MER'  # ['DFC22', 'iSAID', 'MER', 'Vaihingen']
METHOD = 'Ours'  # ['Ours', 'SupOnly', 'FixMatch', 'LSST', 'WSCL', 'MCSS', 'CorrMatch']
# SPLIT = '1-4'  # ['1-4', '1-8', '100', '300'

# Ours
# WEIGHTS = 'checkpoint/DFC22/ResUNet_1-8_42.624_77_200.pth'
# WEIGHTS = 'checkpoint/DFC22/ResUNet_1-4_42.629_125_200.pth'
# WEIGHTS = 'checkpoint/MER/ResUNet_1-8_57.753_137_200.pth'
# WEIGHTS = 'checkpoint/MER/ResUNet_1-4_60.148_157_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/ResUNet_1-8_74.719_196_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/ResUNet_1-4_75.570_182_200.pth'
# WEIGHTS = 'checkpoint/iSAID/ResUNet_100_70.213_43_200.pth'
# WEIGHTS = 'checkpoint/iSAID/ResUNet_300_80.002_126_200.pth'

# MCSS
# WEIGHTS = 'checkpoint/DFC22/MCSS/ResUNet_1-8_41.962_112_200.pth'
# WEIGHTS = 'checkpoint/DFC22/MCSS/ResUNet_1-4_42.378_104_200.pth'
# WEIGHTS = 'checkpoint/MER/MCSS/ResUNet_1-8_56.917_119_200.pth'
# WEIGHTS = 'checkpoint/MER/MCSS/ResUNet_1-4_58.430_72_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/MCSS/ResUNet_1-8_73.844_118_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/MCSS/ResUNet_1-4_75.230_155_200.pth'
# WEIGHTS = 'checkpoint/iSAID/MCSS/ResUNet_100_68.007_125_200.pth'
# WEIGHTS = 'checkpoint/iSAID/MCSS/ResUNet_300_78.596_166_200.pth'

# WSCL
# WEIGHTS = 'checkpoint/DFC22/WSCL/ResUNet_1-8_41.221_180_200.pth'
# WEIGHTS = 'checkpoint/DFC22/WSCL/ResUNet_1-4_41.686_79_200.pth'
# WEIGHTS = 'checkpoint/MER/WSCL/ResUNet_1-8_56.234_148_200.pth'
# WEIGHTS = 'checkpoint/MER/WSCL/ResUNet_1-4_58.667_164_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/WSCL/ResUNet_1-8_74.312_173_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/WSCL/ResUNet_1-4_75.216_187_200.pth'
# WEIGHTS = 'checkpoint/iSAID/WSCL/ResUNet_100_65.703_143_200.pth'
# WEIGHTS = 'checkpoint/iSAID/WSCL/ResUNet_300_77.851_187_200.pth'

# LSST
# WEIGHTS = 'checkpoint/DFC22/LSST/ResUNet_1-8_40.190_110_200.pth'
# WEIGHTS = 'checkpoint/DFC22/LSST/ResUNet_1-4_40.221_65_200.pth'
# WEIGHTS = 'checkpoint/MER/LSST/ResUNet_1-8_54.167_131_200.pth'
# WEIGHTS = 'checkpoint/MER/LSST/ResUNet_1-4_55.317_150_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/LSST/ResUNet_1-8_73.831_194_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/LSST/ResUNet_1-4_74.669_146_200.pth'
# WEIGHTS = 'checkpoint/iSAID/LSST/ResUNet_100_61.440_169_200.pth'
# WEIGHTS = 'checkpoint/iSAID/LSST/ResUNet_300_75.098_161_200.pth'

# RanPaste
# WEIGHTS = 'checkpoint/DFC22/RanPaste/ResUNet_1-8_39.470_154_200.pth'
# WEIGHTS = 'checkpoint/DFC22/RanPaste/ResUNet_1-4_40.661_121_200.pth'
# WEIGHTS = 'checkpoint/MER/RanPaste/ResUNet_1-8_54.764_109_200.pth'
# WEIGHTS = 'checkpoint/MER/RanPaste/ResUNet_1-4_56.365_176_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/RanPaste/ResUNet_1-8_73.917_161_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/RanPaste/ResUNet_1-4_75.011_199_200.pth'
# WEIGHTS = 'checkpoint/iSAID/RanPaste/ResUNet_100_68.589_134_200.pth'
# WEIGHTS = 'checkpoint/iSAID/RanPaste/ResUNet_300_77.387_192_200.pth'

# ICNet
# WEIGHTS = 'checkpoint/DFC22/ICNet/ResUNet_1-8_39.388_160_200.pth'
# WEIGHTS = 'checkpoint/DFC22/ICNet/ResUNet_1-4_40.349_130_200.pth'
# WEIGHTS = 'checkpoint/MER/ICNet/ResUNet_1-8_55.508_144_200.pth'
# WEIGHTS = 'checkpoint/MER/ICNet/ResUNet_1-4_57.542_195_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/ICNet/ResUNet_1-8_74.306_146_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/ICNet/ResUNet_1-4_74.851_198_200.pth'
# WEIGHTS = 'checkpoint/iSAID/ICNet/ResUNet_100_66.166_192_200.pth'
# WEIGHTS = 'checkpoint/iSAID/ICNet/ResUNet_300_76.374_95_200.pth'

# Fixmatch
# WEIGHTS = 'checkpoint/DFC22/Fixmatch/ResUNet_1-8_38.086_147_200.pth'
# WEIGHTS = 'checkpoint/DFC22/Fixmatch/ResUNet_1-4_40.504_134_200.pth'
# WEIGHTS = 'checkpoint/MER/Fixmatch/ResUNet_1-8_54.957_133_200.pth'
# WEIGHTS = 'checkpoint/MER/Fixmatch/ResUNet_1-4_57.590_64_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/Fixmatch/ResUNet_1-8_72.983_68_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/Fixmatch/ResUNet_1-4_74.468_134_200.pth'
# WEIGHTS = 'checkpoint/iSAID/Fixmatch/ResUNet_100_60.413_21_200.pth'
# WEIGHTS = 'checkpoint/iSAID/Fixmatch/ResUNet_300_76.253_90_200.pth'

# SupOnly
# WEIGHTS = 'checkpoint/DFC22/sup_only/ResUNet_1-8_36.674_119_200.pth'
# WEIGHTS = 'checkpoint/DFC22/sup_only/ResUNet_1-4_37.936_194_200.pth'
# WEIGHTS = 'checkpoint/MER/sup_only/ResUNet_1-8_53.415_49_200.pth'
# WEIGHTS = 'checkpoint/MER/sup_only/ResUNet_1-4_55.673_93_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/sup_only/ResUNet_1-8_71.848_197_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/sup_only/ResUNet_1-4_73.409_166_200.pth'
# WEIGHTS = 'checkpoint/iSAID/sup_only/ResUNet_100_57.503_55_200.pth'
# WEIGHTS = 'checkpoint/iSAID/sup_only/ResUNet_300_74.762_82_200.pth'

def DFC22():
    class_names = ['Urban fabric', 'Industrial', 'Mine', 'Artificial', 'Arable', 'Permanent crops',
                   'Pastures', 'Forests', 'Herbaceous', 'Open spaces', 'Wetlands', 'Water']

    palette = [[219, 95, 87], [219, 151, 87], [219, 208, 87], [173, 219, 87], [117, 219, 87], [123, 196, 123],
               [88, 177, 88], [0, 128, 0], [88, 176, 167], [153, 93, 19], [87, 155, 219], [0, 98, 255],[0, 0, 0]]

    return class_names, palette

def Vaihingen():
    class_names = ['Impervious_Surface', 'Building', 'Low_Vegetation', 'Tree', 'Car']

    palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],[255, 255, 0], [255, 0, 0]]

    return class_names, palette

def iSAID():
    class_names = ['Ship', 'Storage_Tank', 'Baseball_Diamond', 'Tennis_Court', 'Basketball_Court',
                   'Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
                   'Swimming_Pool', 'Roundabout','Soccer_Ball_Field', 'Plane', 'Harbor']

    palette = [[0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127], [0, 63, 191],
               [0, 63, 255], [0, 127, 63], [0, 127, 127], [0, 0, 127], [0, 0, 191],
               [0, 0, 255], [0, 191, 127], [0, 127, 191], [0, 127, 255], [0, 100, 155],[0,0,0]]

    return class_names, palette

def MARS():
    class_names = ['Martian Soil', 'Sands', 'Gravel', 'Bedrock', 'Rocks',
                   'Tracks', 'Shadows', 'Unknown', 'Background']

    palette = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
               [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [0, 0, 0]]

    return class_names, palette

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
# semi-supervised settings
parser.add_argument('--config', type=str, default='configs/' + DATASET + '.yaml')
parser.add_argument('--save-path', type=str, default='Visual_result/' + DATASET + '/' + METHOD)

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def main(args, cfg):
    
    create_path(args.save_path)

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', None)
    valloader = DataLoader(valset, batch_size=int(cfg['batch_size'] / 2), shuffle=False,
                           pin_memory=False, num_workers=cfg['num_workers'], drop_last=False)

    print("v ==> ", len(valset))  # 153
    # print("u:l:v ==> ", len(trainloader_u), len(trainloader_l), len(valloader)) # 67 67 20

    model = init_basic_elems(cfg)
    # model, memory = init_basic_elems(cfg) # for MCSS
    print('\nParams: %.1fM' % count_params(model))
    # print(model)
    
    eva_l(model, valloader, cfg)
    # eva_l(model, valloader, cfg, memory) # for MCSS

def init_basic_elems(cfg):
    model_zoo = {  # 'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2,
        'ResUNet': se_resnext50_32x4d}
    if cfg['model'] == 'ResUNet':
        model = model_zoo['ResUNet'](cfg['nclass'], None)
        pretrained_dict = torch.load(WEIGHTS)
        my_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_dict}
        my_dict.update(pretrained_dict)
        model.load_state_dict(my_dict)
    else:
        model = model_zoo[args.model](args.backbone, cfg['nclass'])

    model = nn.DataParallel(model).cuda()

    return model

# def init_basic_elems(cfg): # for MCSS 
#     model_zoo = {  # 'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2,
#         'ResUNet': se_resnext50_32x4d}
#     if cfg['model'] == 'ResUNet':
#         model = model_zoo['ResUNet'](cfg['nclass'], None)
#         pretrained_dict = torch.load(WEIGHTS)
#         memory = pretrained_dict['class_momory'] # for MCSS
#         my_dict = model.state_dict()
#         pretrained = {k: v for k, v in pretrained_dict['state_dict'].items() if k in my_dict} # for MCSS
#         my_dict.update(pretrained)
#         model.load_state_dict(my_dict)
#     else:
#         model = model_zoo[args.model](args.backbone, cfg['nclass'])

#     model = nn.DataParallel(model).cuda() 

#     return model, memory

def eva_l(model, valloader, cfg, memory=None, output_csv_path='Visual_result/' + DATASET + '/' + METHOD +'/evaluation_result.csv'):
    set_to_eval(model)  # 设置为评估模式
    metric = meanIOU(num_classes=cfg['nclass'])
    tbar = tqdm(valloader)
    
    # 打开 CSV 文件进行写入
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 写入表头：文件名和每个类别的像素数量
        header = ['filename'] + [f'class_{i}' for i in range(cfg['nclass'])] + ['total']
        writer.writerow(header)

        with torch.no_grad():
            for batch_idx, (img_batch, mask_batch, id_batch) in enumerate(tbar):
                img_batch = img_batch.cuda()
                # pre_batch = model(img_batch)
                pre_batch = torch.argmax(pre_batch['out'], dim=1).cpu().numpy()  # 预测结果
                pre_batch = torch.argmax(pre_batch, dim=1).cpu().numpy()  # 预测结果
                
                # pre_batch = model(img_batch, memory)  # for_MCSS
                # pre_batch = torch.argmax(pre_batch, dim=1).cpu().numpy()  # 预测结果

                for i in range(len(img_batch)):
                    pred = pre_batch[i]
                    mask = mask_batch[i].numpy()  # 获取对应的mask
                    parts = id_batch[i].split(' ')
                    
                    if len(parts) > 1:
                        labels_part = parts[1]
                        labels_content = labels_part[len('labels/'):]

                        print(f"Processing {labels_content}...\n")
                    else:
                        labels_content = "No_labels"
                    
                    # 统计每个类别的正确像素数量
                    class_counts = [np.sum((pred == j) & (mask == j)) for j in range(cfg['nclass'])]  # 对每个类别进行正确像素统计
                    total_pixels = np.sum(class_counts)  # 计算所有类别的正确像素总和
                    
                    # 写入当前图像的统计数据
                    row = [labels_content] + class_counts + [total_pixels]
                    writer.writerow(row)
                    
                    if DATASET == "iSAID":
                        mask_255_positions = (mask == 255)
                        pred[mask_255_positions] = 15

                    # 保存预测结果为图像
                    result = Image.fromarray(out_to_rgb(pred))
                    result.save(f'Visual_result/{DATASET}/{METHOD}/{labels_content}')
                
                # break;

                # 计算 mIOU 和 IOU
                metric.add_batch(pre_batch, mask_batch.numpy())
                IOU, mIOU = metric.evaluate()
                tbar.set_description(f'mIOU: {mIOU * 100.0:.2f}')

    # 输出最终的 mIOU 和 IOU
    print(f'***** Evaluation ***** >>>> meanIOU: {mIOU*100:.4f}')
    print(f'***** ClassIOU ***** >>>> \n{IOU*100}')

def out_to_rgb(out_index):
    if DATASET == "Vaihingen":
        vai_class, color_map = Vaihingen()
    elif DATASET == "DFC22":  
        DFC22_class, color_map = DFC22()
    elif DATASET == "iSAID":     
        isaid_class, color_map = iSAID()
    elif DATASET == "MER":     
        mer_class, color_map = MARS()
    
    colormap = np.array(color_map)
    rgb_img = colormap[np.array(out_index)].astype(np.uint8)
    return rgb_img

if __name__ == '__main__':
    args = parser.parse_args()

    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    print('{}\n'.format(pprint.pformat(cfg)))

    main(args, cfg)