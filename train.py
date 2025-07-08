import argparse
import os
import pprint
import yaml
from torch import nn
import torch.nn.functional as F

from dataset.semi import SemiDataset
from utils.utils import *
from model.SEUNet import se_resnext50_32x4d

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# import torch.multiprocessing
# from torch.cuda.amp import autocast
# from torch.cuda.amp import grad_scaler

from tensorboardX import SummaryWriter
from utils.custom_eval_model import set_to_eval

writer = SummaryWriter(log_dir='logs')

# torch.multiprocessing.set_sharing_strategy('file_system') # 使用文件系统作为数据共享的策略。
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

seed = 1234
set_random_seed(seed)

DATASET = 'MER'  # ['DFC22', 'iSAID', 'MER', 'Vaihingen']
SPLIT = '1-8'  # ['1-4', '1-8', '100', '300']

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')   
# semi-supervised settings
parser.add_argument('--config', type=str, default='configs/' + DATASET + '.yaml')
parser.add_argument('--labeled-id-path', type=str, default='partitions/' + DATASET + '/' + SPLIT + '/labeled.txt')
# parser.add_argument('--labeled-id-path', type=str, default='partitions/' + DATASET + '/' + SPLIT + '/labeled_srcs.txt')
parser.add_argument('--unlabeled-id-path', type=str, default='partitions/' + DATASET + '/' + SPLIT + '/unlabeled.txt')
parser.add_argument('--save-path', type=str, default='checkpoint/' + DATASET)


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args, cfg):
    create_path(args.save_path)

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs'])
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none')
    criterion_kl = nn.KLDivLoss(reduction='none')

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', None, None, None, None)
    valloader = DataLoader(valset, batch_size=int(cfg['batch_size'] / 2), shuffle=False,
                           pin_memory=False, num_workers=cfg['num_workers'], drop_last=False)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path,
                            None, cfg['BMG'])
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path,
                             nsample=len(trainset_u.ids))

    trainloader_u = DataLoader(trainset_u, batch_size=int(cfg['batch_size'] / 2), shuffle=True,
                               pin_memory=False, num_workers=cfg['num_workers'], drop_last=True)
    trainloader_l = DataLoader(trainset_l, batch_size=int(cfg['batch_size'] / 2), shuffle=True,
                               pin_memory=False, num_workers=cfg['num_workers'], drop_last=True)
    print("u:l:v ==> ", len(trainset_u), len(trainset_l), len(valset))  # 537 537 153
    # print("u:l:v ==> ", len(trainloader_u), len(trainloader_l), len(valloader)) # 67 67 20

    model, optimizer = init_basic_elems(cfg, trainloader_u)
    print('\nParams: %.1fM' % count_params(model))
    # print(model)

    train_basematch(model, trainloader_l, trainloader_u, valloader, criterion, criterion_u, criterion_kl, optimizer, cfg)


def init_basic_elems(cfg, trainloader_u):
    model_zoo = {  # 'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2,
        'ResUNet': se_resnext50_32x4d}
    if cfg['model'] == 'ResUNet':
        model = model_zoo['ResUNet'](cfg['nclass'], None)
        pretrained_dict = torch.load(
            'pretrained/se_resnext50_32x4d-a260b3a4.pth')
        my_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_dict}
        my_dict.update(pretrained_dict)
        model.load_state_dict(my_dict)
    else:
        model = model_zoo[args.model](args.backbone, cfg['nclass'])

    head_lr_multiple = cfg['lr_multi']

    if cfg['model'] == 'deeplabv2':
        assert cfg['backbone'] == 'resnet101'
        head_lr_multiple = 1.0

    #     optimizer = optim.SGD([# {'params': model.backbone.parameters(), 'lr': args.lr},
    #                      {'params': [param for name, param in model.named_parameters()
    #                                  if 'backbone' not in name],
    #                       'lr': cfg['lr'] * head_lr_multiple}],
    #                     lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    #     optimizer = optim.Adam([ # {'params': model.backbone.parameters(), 'lr': args.lr},
    #                         {'params': [param for name, param in model.named_parameters()
    #                                     if 'backbone' not in name],
    #                          'lr': cfg['lr'] * head_lr_multiple}],
    #                         lr=cfg['lr'],  weight_decay=1e-4)
    optimizer = optim.AdamW([
        # {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': [param for name, param in model.named_parameters()
                    if 'backbone' not in name],
         'lr': cfg['lr'] * head_lr_multiple}],
        lr=cfg['lr'],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False
    )
        
    model = nn.DataParallel(model).cuda()

    return model, optimizer


def train_basematch(model, trainloader_l, trainloader_u, valloader, criterion, criterion_u, criterion_kl, optimizer, cfg):
    iters = 0
    total_iters = len(trainloader_u) * cfg['epochs']

    previous_best = 0.0
    previous_best_iou = 0.0
    previous_best_epoch = 0
    weight_u = cfg['lamda']
    threshold = cfg['thresh_init']  # 初始阈值

    for epoch in range(cfg['epochs']):

        threshold = calculate_decay_threshold(cfg['thresh_init'], cfg['thresh_final'], 0, cfg['epochs'], epoch)
        threshold_mask = calculate_decay_threshold(cfg['thresh_mask_init'], cfg['thresh_mask_final'], 0, cfg['epochs'], epoch)

        print("\n==> Epoch %i, learning rate = %.5f\t threshold = %.5f\t previous best = %.2f \n %s" %
              (epoch, optimizer.param_groups[0]["lr"], threshold, previous_best, previous_best_iou))

        total_loss, total_loss_l, total_loss_u = 0.0, 0.0, 0.0
        total_loss_mask, total_loss_w_fp = 0.0, 0.0
        total_loss_kl = 0.0

        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))

        for i, ((img, mask, img2, mask2),
                (img_u_w, img2_u_w, cutmix_box, img_u_s1, _, _, _, _, masked_img)) in enumerate(tbar):
            img, mask, img2, mask2 = img.cuda(), mask.cuda(), img2.cuda(), mask2.cuda(),
            img_u_w, img2_u_w, cutmix_box, img_u_s1 = img_u_w.cuda(), img2_u_w.cuda(), cutmix_box.cuda(), img_u_s1.cuda()
            masked_img = masked_img.cuda()
            # ignore_mask = ignore_mask.cuda()
            # print(cutmix_box.shape) # torch.Size([8, 320, 320])

            num_lb, num_ulb = img.shape[0], img_u_w.shape[0]  # 8 8

            with torch.no_grad():
                set_to_eval(model)
                res_u_w_dict = model(torch.cat((img_u_w, img2_u_w)))
                preds_u_w = res_u_w_dict['out']
                pred_u_w, pred2_u_w = preds_u_w.split([num_ulb, num_ulb])
                prob_u_w = pred_u_w.softmax(dim=1)
                conf_u_w, mask_u_w = prob_u_w.max(dim=1)
                prob2_u_w = pred2_u_w.softmax(dim=1)
                conf2_u_w, mask2_u_w = prob2_u_w.max(dim=1) # torch.Size([8, 320, 320]) torch.Size([8, 320, 320])
                # print(conf2_u_w.shape, mask2_u_w.shape)
                mask_u_w_mix, conf_u_w_mix = mask_u_w.clone(), conf_u_w.clone()
                conf_u_w_mix[cutmix_box == 1] = conf2_u_w[cutmix_box == 1]
                mask_u_w_mix[cutmix_box == 1] = mask2_u_w[cutmix_box == 1]
                # pred_u_w_fp = preds_fp[num_ulb:]
                # print(pred_u_w_fp.shape)

            model.train()

            preds_dict = model(torch.cat((img, img_u_s1, masked_img)))
            preds = preds_dict['out']
            pred, pred_u_s1, pred_mask = preds.split([num_lb, num_ulb, num_ulb])
            
            pred_dict = model(img_u_w, need_fp = True, ddp = cfg['DDP'])
            pred_u_w_fp = pred_dict['out_fp']
 
            # label
            loss_l = criterion(pred, mask)

            # un_label
            conf_filter_u_w_mix = (conf_u_w_mix >= threshold)
            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_mix)
            loss_u_s1 = loss_u_s1 * conf_filter_u_w_mix
            loss_u_s1 = torch.mean(loss_u_s1)
            
            # mask
            conf_filter_u_w = (conf_u_w >= threshold_mask)
            loss_mask = criterion_u(pred_mask, mask_u_w)
            loss_mask = loss_mask * conf_filter_u_w
            loss_mask = torch.mean(loss_mask)
            
            # fp
            if cfg['dataset'] == 'DFC22' and SPLIT == '1-8':
                conf_filter_fp = (conf_u_w >= 0.968) # for DFC22 1-8 
            else:
                conf_filter_fp = (conf_u_w >= threshold)
            loss_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_fp = loss_fp * conf_filter_fp
            loss_fp = torch.mean(loss_fp)
            
            if cfg['dataset'] == 'DFC22' and SPLIT == '1-8':
                loss_u = (0.4 * loss_u_s1 + 0.4 * loss_mask + 0.2 * loss_fp)
            elif cfg['dataset'] == 'MER' and SPLIT == '1-8':
                loss_u = (0.4 * loss_u_s1 + 0.4 * loss_mask + 0.2 * loss_fp)
            else:
                loss_u = (loss_u_s1 + loss_mask + loss_fp) / 3.0  #for others
            loss = 0.5 * (loss_l + weight_u * loss_u)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_l += loss_l.item()
            total_loss_u += loss_u_s1.item()
            total_loss_mask += loss_mask.item()
            total_loss_w_fp += loss_fp.item()
            # total_loss_kl += loss_u_kl.item()
            
            # lr_scheduler.step()
            # iters = epoch * len(trainloader_u) + i
            # lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr * cfg['lr_multi']

            tbar.set_description('Total loss: {:.3f}, Loss l: {:.3f}, loss_u_s1: {:.3f} '
                                     'loss_mask: {:.3f}, loss_fp: {:.3f}, loss_kl: {:.3f}'.format(
                total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u / (i + 1), 
                total_loss_mask / (i + 1), total_loss_w_fp / (i + 1), total_loss_kl / (i + 1)))
            tbar.update(1)

        writer.add_scalar('loss', total_loss, global_step=epoch)
        writer.add_scalar('Threshold', threshold, global_step=epoch)
        writer.add_scalar('Threshold_mask', threshold_mask, global_step=epoch)
        writer.add_scalar('loss_l', total_loss_l, global_step=epoch)
        writer.add_scalar('loss_u', total_loss_u, global_step=epoch)
        writer.add_scalar('loss_m', total_loss_mask, global_step=epoch)
        writer.add_scalar('loss_f', total_loss_w_fp, global_step=epoch)
        

        tbar.close()

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        if (epoch + 1) % 1 == 0:
            metric = meanIOU(num_classes=cfg['nclass'])
            tbar = tqdm(valloader)

            with torch.no_grad():
                for img, mask, _ in tbar:
                    set_to_eval(model)
                    img = img.cuda()
                    pred_dict = model(img)
                    pred = torch.argmax(pred_dict['out'], dim=1)

                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()
                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            mIOU *= 100.0
            IOU *= 100

            print('***** Evaluation ***** >>>> meanIOU: {:.4f} \n'.format(mIOU))
            print('***** ClassIOU ***** >>>> \n{}\n'.format(IOU))
            writer.add_scalar('mIOU', mIOU, global_step=epoch)

            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(os.path.join("checkpoint", cfg['dataset'], '%s_%s_%.3f_%.f_%.f.pth' % (
                        cfg['model'], SPLIT, previous_best, previous_best_epoch + 1, cfg['epochs'])))
                previous_best = mIOU
                previous_best_iou = IOU
                previous_best_epoch = epoch
                torch.save(model.module.state_dict(), os.path.join("checkpoint", cfg['dataset'],
                                                                   '%s_%s_%.3f_%.f_%.f.pth' % (
                                                                        cfg['model'], SPLIT, mIOU, epoch + 1, cfg['epochs'])))

            torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parser.parse_args()

    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    print('{}\n'.format(pprint.pformat(cfg)))

    main(args, cfg)

    writer.close()