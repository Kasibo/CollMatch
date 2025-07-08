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

# torch.multiprocessing.set_sharing_strategy('file_system') # 使用文件系统作为数据共享的策略。
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

DATASET = 'Vaihingen'  # ['DFC22', 'iSAID', 'MER', 'Vaihingen']
# SPLIT = '1-4'  # ['1-4', '1-8', '100', '300'

# Ours
# WEIGHTS = 'checkpoint/DFC22/ResUNet_1-8_42.624_77_200.pth'
# WEIGHTS = 'checkpoint/DFC22/ResUNet_1-4_42.629_125_200.pth'
# WEIGHTS = 'checkpoint/MER/ResUNet_1-8_57.753_137_200.pth'
# WEIGHTS = 'checkpoint/MER/ResUNet_1-4_60.148_157_200.pth'
# WEIGHTS = 'checkpoint/Vaihingen/ResUNet_1-8_74.719_196_200.pth'
WEIGHTS = 'checkpoint/Vaihingen/ResUNet_1-4_75.570_182_200.pth'
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

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
# semi-supervised settings
parser.add_argument('--config', type=str, default='configs/' + DATASET + '.yaml')

def main(args, cfg):

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


def eva_l(model, valloader, cfg, memory = None):
    metric = meanIOU(num_classes=cfg['nclass'])

    tbar = tqdm(valloader)

    with torch.no_grad():
        set_to_eval(model)
        for img, mask, _ in tbar:
            img = img.cuda()
            pred_dict = model(img)
            pred = torch.argmax(pred_dict['out'], dim=1)
            
            # pred_dict = model(img, memory)  # for_MCSS
            # pred = torch.argmax(pred_dict, dim=1) # for_MCSS
            
            # pred_dict = model(img) # for_others
            # pred = torch.argmax(pred_dict, dim=1) # for_others
            
            metric.add_batch(pred.cpu().numpy(), mask.numpy())
            IOU, mIOU = metric.evaluate()
            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

    mIOU *= 100.0
    IOU *= 100

    print('***** Evaluation ***** >>>> meanIOU: {:.4f} \n'.format(mIOU))
    print('***** ClassIOU ***** >>>> \n{}\n'.format(IOU))


if __name__ == '__main__':
    args = parser.parse_args()

    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    print('{}\n'.format(pprint.pformat(cfg)))

    main(args, cfg)