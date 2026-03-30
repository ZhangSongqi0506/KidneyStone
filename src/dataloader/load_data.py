# -*- coding: utf-8 -*-
# Time    : 2023/10/30 16:03
# Author  : fanc
# File    : load_data.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import json
from skimage.transform import resize
import SimpleITK as sitk
from scipy.ndimage import zoom
from collections import defaultdict
from skimage.morphology import dilation, ball, closing
from scipy.ndimage import gaussian_filter
import pandas as pd

def split_data_with_inner_folds(data_dir, infos_name, filter_volume, rate=0.8, n_inner_folds=4):
    """
    分层分割数据集，并在训练集内部分出4折用于交叉验证
    
    Args:
        data_dir: 数据目录路径
        infos_name: 数据信息文件名  
        filter_volume: 体积过滤阈值
        rate: 训练集分割比例
        n_inner_folds: 内部分折数
    
    Returns:
        train_infos: 训练集数据信息列表
        test_infos: 测试集数据信息列表
        inner_folds: 训练集内部的4折分割 [fold1, fold2, fold3, fold4]
    """
    # 读取数据信息文件
    with open(os.path.join(data_dir, infos_name), 'r', encoding='utf-8') as f:
        infos = json.load(f)

    # 过滤数据
    infos = list(filter(lambda x: x['volume'] >= filter_volume, infos))
    
    # 按类别存储数据
    class_data = defaultdict(list)
    for info in infos:
        label = info['label']
        class_data[label].append(info)

    # 初始化训练集和测试集
    train_infos = []
    test_infos = []

    # 对每个类别进行分层抽样
    for label, data in class_data.items():
        random.seed(1900)
        random.shuffle(data)
        
        num_samples = len(data)
        train_num = int(rate * num_samples)
        
        # 外层分割
        train_infos.extend(data[:train_num])
        test_infos.extend(data[train_num:])

    # 在训练集内部创建4折
    inner_folds = create_simple_folds(train_infos, n_folds=n_inner_folds)
    
    return train_infos, test_infos, inner_folds

def create_simple_folds(data, n_folds=4):
    """
    简单将数据分成n折
    """
    random.seed(1900)
    random.shuffle(data)
    
    fold_size = len(data) // n_folds
    folds = []
    
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(data)
        folds.append(data[start_idx:end_idx])
    
    return folds

def split_data(data_dir, infos_name, filter_volume, rate=0.8):
    """
    分层分割数据集，确保训练集和测试集中各类别比例一致
    
    Args:
        data_dir: 数据目录路径
        infos_name: 数据信息文件名
        filter_volume: 体积过滤阈值，只保留体积大于此值的样本
        rate: 训练集分割比例，默认0.8(80%)
    
    Returns:
        train_infos: 训练集数据信息列表
        test_infos: 测试集数据信息列表
    """
    # 读取数据信息文件
    with open(os.path.join(data_dir, infos_name), 'r', encoding='utf-8') as f:
        infos = json.load(f)

    # 过滤数据：只保留体积大于等于阈值的样本
    # 这通常用于去除太小或不重要的病灶
    infos = list(filter(lambda x: x['volume'] >= filter_volume, infos))
    
    # 创建一个字典，用于按类别存储数据
    # 键：类别标签，值：该类别的数据样本列表
    class_data = defaultdict(list)
    for info in infos:
        label = info['label']  # 假设数据集中每个样本都有'label'字段表示类别
        class_data[label].append(info)

    # 初始化训练集和测试集
    train_infos = []
    test_infos = []

    # 对每个类别进行分层抽样
    for label, data in class_data.items():
        # 设置随机种子确保每次分割结果一致（可重复性）
        random.seed(1900)
        #random.seed(190)
        # 打乱当前类别的数据顺序
        random.shuffle(data)
        
        # 计算当前类别的样本数量
        num_samples = len(data)
        # 计算当前类别应该分配到训练集的样本数量
        train_num = int(rate * num_samples)
        
        # 将数据分割到训练集和测试集
        train_infos.extend(data[:train_num])  # 前train_num个样本到训练集
        test_infos.extend(data[train_num:])   # 剩余样本到测试集

    return train_infos, test_infos
# def split_data(data_dir, infos_name, filter_volume, rate=0.8):
#     """
#     患者级别的分层分割数据集，确保：
#     1. 同一患者的所有样本在同一个集合中
#     2. 训练集和测试集中各类别比例与原始数据一致
    
#     Args:
#         data_dir: 数据目录路径
#         infos_name: 数据信息文件名
#         filter_volume: 体积过滤阈值，只保留体积大于此值的样本
#         rate: 训练集分割比例，默认0.8(80%)
    
#     Returns:
#         train_infos: 训练集数据信息列表
#         test_infos: 测试集数据信息列表
#     """
#     # 读取数据信息文件
#     with open(os.path.join(data_dir, infos_name), 'r', encoding='utf-8') as f:
#         infos = json.load(f)

#     # 过滤数据：只保留体积大于等于阈值的样本
#     infos = list(filter(lambda x: x['volume'] >= filter_volume, infos))
    
#     # 第一步：按患者分组，收集每个患者的所有样本
#     patient_data = defaultdict(list)
#     for info in infos:
#         pid = info['pid']  # 使用患者ID作为分组键
#         patient_data[pid].append(info)
    
#     # 第二步：按患者标签分组，用于分层抽样
#     # 假设同一患者的所有样本标签相同
#     patient_by_class = defaultdict(list)
#     for pid, samples in patient_data.items():
#         label = samples[0]['label']  # 取第一个样本的标签作为患者标签
#         patient_by_class[label].append(pid)  # 存储患者ID而非具体样本

#     # 初始化训练集和测试集
#     train_infos = []
#     test_infos = []

#     # 第三步：对每个类别进行患者级别的分层抽样
#     for label, patient_ids in patient_by_class.items():
#         # 设置随机种子确保可重复性
#         random.seed(1900)
#         # 打乱当前类别的患者顺序
#         random.shuffle(patient_ids)
        
#         # 计算当前类别应该分配到训练集的患者数量
#         num_patients = len(patient_ids)
#         train_num = int(rate * num_patients)
        
#         # 选择训练集患者和测试集患者
#         train_patients = patient_ids[:train_num]
#         test_patients = patient_ids[train_num:]
        
#         # 第四步：根据患者ID将样本分配到训练集和测试集
#         # 训练集：该患者的所有样本加入训练集
#         for pid in train_patients:
#             train_infos.extend(patient_data[pid])
            
#         # 测试集：该患者的所有样本加入测试集
#         for pid in test_patients:
#             test_infos.extend(patient_data[pid])

#     # # 验证分割结果
#     # _validate_split_results(train_infos, test_infos)
    
#     return train_infos, test_infos
# class MyDataset(Dataset):
#     def __init__(self, data_dir, infos, input_size, phase='train', task=[0, 1]):
#         '''
#         task: 0 :seg,  1 :cla
#         '''
#         task = [int(i) for i in re.findall('\d', str(task))]
#         img_dir = os.path.join(data_dir, 'imgs_nii')
#         mask_dir = os.path.join(data_dir, 'mask_nii')
#         self.seg = False
#         self.cla = False
#         if 0 in task:
#             self.seg = True
#         if 1 in task:
#             self.cla = True
#
#         self.input_size = tuple([int(i) for i in re.findall('\d+', str(input_size))])
#         self.img_dir = img_dir
#         if self.cla:
#             self.labels = [i['label'] for i in infos]
#         if self.seg:
#             self.mask_dir = mask_dir
#         # self.labels = [[1, 0] if int(i['label']) == 1 else [0, 1] for i in infos]
#
#         self.ids = [i['id'] for i in infos]
#         self.phase = phase
#         self.labels = torch.tensor(self.labels, dtype=torch.float)
#
#     def __len__(self):
#         return len(self.ids)
#
#     def __getitem__(self, i):
#         img = sitk.ReadImage(os.path.join(self.img_dir, f"{self.ids[i]}.nii.gz"))
#         if self.seg:
#             mask = sitk.ReadImage(os.path.join(self.mask_dir, f"{self.ids[i]}-mask.nii.gz"))
#         else:
#             mask = None
#         if self.phase == 'train':
#             img, mask = self.train_preprocess(img, mask)
#         else:
#             img, mask = self.val_preprocess(img, mask)
#         if self.cla:
#             label = self.labels[i]
#         else:
#             label = 1
#
#         img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
#         if self.seg:
#             mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
#         if self.cla:
#             label = self.labels[i].unsqueeze(0)
#
#         return img, mask, label
#
#     def train_preprocess(self, img, mask):
#         img, mask = self.resample(itkimage=img, itkmask=mask)
#         # mask = self.resample(mask)
#         # print(img.shape, mask.shape)
#         if self.seg:
#             assert img.shape == mask.shape, "img and mask shape not match"
#             # img, mask = self.crop(img, mask)
#         img = self.normalize(img)
#         img, mask = self.resize(img, mask)
#
#         return img, mask
#     def val_preprocess(self, img, mask):
#         img, mask = self.resample(img, mask)
#         # mask = self.resample(mask)
#         if self.seg:
#             assert img.shape == mask.shape, "img and mask shape not match"
#         # img, mask = self.crop(img, mask)
#         img = self.normalize(img)
#         img, mask = self.resize(img, mask)
#
#         return img, mask
#
#     def crop(self, img, mask):
#         crop_img = img
#         crop_mask = mask
#         # amos kidney mask
#         crop_mask[crop_mask == 2] = 1
#         crop_mask[crop_mask != 1] = 0
#         target = np.where(crop_mask == 1)
#         [d, h, w] = crop_img.shape
#         [max_d, max_h, max_w] = np.max(np.array(target), axis=1)
#         [min_d, min_h, min_w] = np.min(np.array(target), axis=1)
#         [target_d, target_h, target_w] = np.array([max_d, max_h, max_w]) - np.array([min_d, min_h, min_w])
#         z_min = int((min_d - target_d / 2) * random.random())
#         y_min = int((min_h - target_h / 2) * random.random())
#         x_min = int((min_w - target_w / 2) * random.random())
#
#         z_max = int(d - ((d - (max_d + target_d / 2)) * random.random()))
#         y_max = int(h - ((h - (max_h + target_h / 2)) * random.random()))
#         x_max = int(w - ((w - (max_w + target_w / 2)) * random.random()))
#
#         z_min = np.max([0, z_min])
#         y_min = np.max([0, y_min])
#         x_min = np.max([0, x_min])
#
#         z_max = np.min([d, z_max])
#         y_max = np.min([h, y_max])
#         x_max = np.min([w, x_max])
#
#         z_min = int(z_min)
#         y_min = int(y_min)
#         x_min = int(x_min)
#
#         z_max = int(z_max)
#         y_max = int(y_max)
#         x_max = int(x_max)
#         crop_img = crop_img[z_min: z_max, y_min: y_max, x_min: x_max]
#         crop_mask = crop_mask[z_min: z_max, y_min: y_max, x_min: x_max]
#
#         return crop_img, crop_mask
#
#     def resample(self, itkimage, itkmask, new_spacing=[1, 1, 1]):
#         # spacing = itkimage.GetSpacing()
#         img = sitk.GetArrayFromImage(itkimage)
#         if self.seg:
#             mask = sitk.GetArrayFromImage(itkmask)
#         else:
#             mask = None
#         # # MASK 膨胀腐蚀操作
#         # kernel = ball(5)  # 3D球形核
#         # # 应用3D膨胀
#         # dilated_mask = dilation(mask, kernel)
#         # mask = closing(dilated_mask, kernel)
#         # resize_factor = spacing / np.array(new_spacing)
#         # resample_img = zoom(img, resize_factor, order=0)
#         # resample_mask = zoom(mask, resize_factor, order=0, mode='nearest')
#         return np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)
#
#     def normalize(self, img):
#
#         # CT值范围选取
#         min = 0
#         max = 2000
#         img[img < min] = min
#         img[img > max] = max
#
#         # std = np.std(img)
#         # avg = np.average(img)
#         # return (img - avg + std) / (std * 2)
#         return (img - min) / (max - min)
#
#     def resize(self, img, mask):
#         # img = np.transpose(img, (2, 1, 0))
#         # mask = np.transpose(mask, (2, 1, 0))
#         rate = np.array(self.input_size) / np.array(img.shape)
#         try:
#             img = zoom(img, rate.tolist(), order=0)
#             if self.seg:
#                 mask = zoom(mask, rate.tolist(), order=0, mode='nearest')
#         except Exception as e:
#             print(e)
#             img = resize(img, self.input_size)
#             if self.seg:
#                 mask = resize(mask, self.input_size, order=0)
#         # # MASK 膨胀腐蚀操作
#         # kernel = ball(5)  # 3D球形核
#         # # 应用3D膨胀
#         # dilated_mask = dilation(mask, kernel)
#         # mask = closing(dilated_mask, kernel)
#
#         # 高斯滤波去噪
#         # img = gaussian_filter(img, sigma=1)
#         # 中值滤波去噪
#         # from scipy.ndimage import median_filter
#         # img = median_filter(img, size=3)
#
#         return img, mask

class MyDataset(Dataset):
    def __init__(self, data_dir, infos, phase='train'):
        """
        医学影像数据集类
        
        Args:
            data_dir: 数据根目录
            infos: 数据信息列表，包含sid, pid, label等信息
            phase: 阶段，'train' 或 'test'
        """
        # 加载数据集配置文件
        with open('configs/dataset.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 从配置文件中获取各数据路径
        img_dir = config['img_dir']        # 图像文件目录
        mask_dir = config['mask_dir']      # 分割掩码目录
        data_dir = config['data_dir']      # 数据根目录
        clinical_dir = config['clinical_dir']  # 临床数据文件路径

        # 初始化临床数据使用标志
        self.use_clinical = False
        if clinical_dir:
            self.use_clinical = True
            # 读取临床数据Excel文件
            self.clinical = pd.read_excel(os.path.join(data_dir, clinical_dir))
        
        # 设置图像和掩码目录的完整路径
        self.img_dir = os.path.join(data_dir, img_dir)
        self.mask_dir = os.path.join(data_dir, mask_dir)
        
        # 从infos中提取数据标识信息
        self.labels = [i['label'] for i in infos]  # 标签列表
        self.ids = [i['sid'] for i in infos]       # 样本ID列表
        self.pids = [i['pid'] for i in infos]      # 患者ID列表
        self.phase = phase                         # 训练或测试阶段
        
        # 将标签转换为tensor
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        """返回数据集大小"""
        return len(self.ids)

    def __getitem__(self, i):
        """
        获取单个数据样本
        
        Returns:
            img: 图像数据 [1, H, W, D]
            mask: 分割掩码 [1, H, W, D] 
            label: 分类标签 [1]
            clinical: 临床数据 [feature_dim] 或 0
        """
        # 读取图像和掩码文件
        img = sitk.ReadImage(os.path.join(self.img_dir, f"{self.ids[i]}.nii.gz"))
        mask = sitk.ReadImage(os.path.join(self.mask_dir, f"{self.ids[i]}.nii.gz"))
        
        # 根据阶段进行不同的预处理
        if self.phase == 'train':
            img, mask = self.train_preprocess(img, mask)
        else:
            img, mask = self.val_preprocess(img, mask)
        
        # 转换为tensor并添加通道维度
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)    # [1, H, W, D]
        mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)    # [1, H, W, D]
        label = self.labels[i].unsqueeze(0)                          # [1]

        # 处理临床数据
        if self.use_clinical:
            pid = self.pids[i]  # 获取患者ID
            # 根据患者ID筛选对应的临床数据
            clinical = self.clinical[self.clinical['pid']==pid].fillna(0)
            # 提取特征值（跳过第一列的pid）
            clinical = torch.tensor(np.array(clinical.values[0][1:], dtype=np.float32), dtype=torch.float32)
        else:
            clinical = 0  # 无临床数据时返回0

        return img, mask, label, clinical

    def train_preprocess(self, img, mask):
        """
        训练阶段预处理
        - 重采样到统一尺寸
        - 标准化图像强度
        """
        img, mask = self.resample(itkimage=img, itkmask=mask)
        img = self.normalize(img)
        return img, mask
    
    def val_preprocess(self, img, mask):
        """
        验证/测试阶段预处理
        - 重采样到统一尺寸  
        - 标准化图像强度
        """
        img, mask = self.resample(img, mask)
        img = self.normalize(img)
        return img, mask

    def resample(self, itkimage, itkmask):
        """
        重采样函数（当前实现仅为格式转换）
        
        Note: 实际应用中这里应该包含空间重采样到统一分辨率
        目前只是简单地将SimpleITK图像转换为numpy数组
        """
        # 从SimpleITK图像获取numpy数组
        img = sitk.GetArrayFromImage(itkimage)   # 图像数据 [H, W, D]
        mask = sitk.GetArrayFromImage(itkmask)   # 掩码数据 [H, W, D]

        return np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)

    def normalize(self, img):
        """
        CT图像标准化
        - 截断CT值到合理范围（-400到2000 HU）
        - 归一化到[0, 1]区间
        
        CT值范围说明:
        - -400 HU: 去除空气和脂肪等极低密度组织
        - 2000 HU: 去除骨骼和金属植入物等极高密度组织
        """
        # 设置CT值范围
        min = -400   # 最小值（HU单位）
        max = 2000   # 最大值（HU单位）
        
        # 截断超出范围的CT值
        img[img < min] = min
        img[img > max] = max

        # 归一化到[0, 1]范围
        return (img - min) / (max - min)


def my_dataloader(data_dir, infos, batch_size=1, shuffle=True, num_workers=0):
    dataset = MyDataset(data_dir, infos)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == '__main__':

    data_dir = r'C:\Users\Asus\Desktop\KidneyStone\data'
    train_info, test_info = split_data(data_dir, rate=0.8)
    train_dataloader = my_dataloader(data_dir, train_info, batch_size=1)
    test_dataloader = my_dataloader(data_dir, test_info, batch_size=1)
    for i, (image, mask, label) in enumerate(train_dataloader):
        print(image, mask.max(), label.shape, label[:, 0].shape)

        new_image = sitk.GetImageFromArray(image.numpy()[0][0])
        new_image.SetSpacing([1, 1, 1])
        sitk.WriteImage(new_image, os.path.join(data_dir, f'{i}.nii.gz'))

        new_mask = sitk.GetImageFromArray(mask.numpy()[0][0])
        new_mask.SetSpacing([1, 1, 1])
        sitk.WriteImage(new_mask, os.path.join(data_dir, f'{i}-mask.nii.gz'))
        break

        # nifti_image = nib.Nifti1Image(image.numpy()[0][0], affine=None)
        # nib.save(nifti_image, os.path.join(data_dir, f'process_img_{i}.nii.gz'))
        # nifti_image = nib.Nifti1Image(mask.numpy()[0][0], affine=None)
        # nib.save(nifti_image, os.path.join(data_dir, f'process_mask_{i}.nii.gz'))
    #

    # for i, (image, mask, label) in enumerate(test_dataloader):
    #     print(i,  image.shape, mask.shape, label)
