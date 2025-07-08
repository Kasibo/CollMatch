import math
import random

import numpy as np
import torch
from PIL import Image

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # if deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

def calculate_decay_threshold(initial_value, final_value, start_epoch, end_epoch, now_epoch):
    if final_value > initial_value:
        raise ValueError("Final value must be less than initial value.")
    total_epochs = end_epoch - start_epoch + 1
    decay_rate = -math.log(final_value / initial_value) / total_epochs
    decay_value = initial_value * math.exp(-decay_rate * (now_epoch - start_epoch + 1))
    return decay_value

class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + 1e-10)
        return iu, np.nanmean(iu)
    
def generate_cutmix_mask(img_size, ratio=2):
    cut_area = img_size[0] * img_size[1] / ratio
    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cut_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)
    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0

    return mask.long()

def generate_unsup_aug_sc(conf_w, mask_w, data_s):
    b, _, im_h, im_w = data_s.shape
    device = data_s.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_data_s.append((data_s[i] * augmix_mask + data_s[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    new_conf_w, new_mask_w, new_data_s = (torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s))

    return new_conf_w, new_mask_w, new_data_s

def generate_unsup_aug_ds(data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_data_s = []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_data_s.append((data_s1[i] * augmix_mask + data_s2[i] * (1 - augmix_mask)).unsqueeze(0))
    new_data_s = torch.cat(new_data_s)

    return new_data_s

def generate_unsup_aug_dc(conf_w, mask_w, data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_data_s.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    new_conf_w, new_mask_w, new_data_s = (torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s))

    return new_conf_w, new_mask_w, new_data_s

def generate_unsup_aug_sdc(conf_w, mask_w, data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        if i % 2 == 0:
            new_data_s.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        else:
            new_data_s.append((data_s2[i] * augmix_mask + data_s1[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))

    new_conf_w, new_mask_w, new_data_s = (torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s))

    return new_conf_w, new_mask_w, new_data_s


def entropy_map(a, dim):
    em = - torch.sum(a * torch.log2(a + 1e-10), dim=dim)
    return em


def self_adaptive_class_mix(selected_classes, w_datas1, w_datas2, s1_datas1, s2_datas1, sup_image, sup_label, p=0.5): # MCSS专用
    if random.random() > p:
        classmix_matric = None
        return w_datas1, w_datas2, s1_datas1, s2_datas1, classmix_matric
    size = sup_label.size()
    classmix_matric = torch.zeros(size).cuda()
    selected_classes = torch.unique(torch.tensor(selected_classes)).cuda()
    included_classes = [cls for cls in torch.unique(sup_label).int() if cls in selected_classes]
    if not included_classes:
        return w_datas1, w_datas2, s1_datas1, s2_datas1, None
    for class_idx in included_classes:
        classmix_matric[(sup_label == class_idx)] = 1
    w_datas1[:, :, classmix_matric == 1] = sup_image[:, classmix_matric == 1]
    w_datas2[:, :, classmix_matric == 1] = sup_image[:, classmix_matric == 1]
    s1_datas1[:, :, classmix_matric == 1] = sup_image[:, classmix_matric == 1]
    s2_datas1[:, :, classmix_matric == 1] = sup_image[:, classmix_matric == 1]
    return w_datas1, w_datas2, s1_datas1, s2_datas1, classmix_matric


