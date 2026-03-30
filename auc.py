import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from src.dataloader.load_data import split_data, my_dataloader
import os.path
import os
from src.models.networks.nets import DoubleFlow, UNETR, KSCNet, HyMNet
from src.models.SegPrompt.unet3d.model import UNet3D
from src.models.SegPrompt.SegMapEncoder import SegPromptBackbone
from src.models.networks.resnet import generate_model
import json
from torchvision.models import resnet34, resnet50, resnet101  # 导入 ResNet 模型


# 1. 加载模型
def load_model(model, checkpoint_path, multi_gpu=False):
    """
    通用加载模型函数。

    :param model: 要加载状态字典的PyTorch模型。
    :param checkpoint_path: 模型权重文件的路径。
    :param multi_gpu: 布尔值，指示是否使用多GPU加载模型。
    :return: 加载了权重的模型。
    """
    # 加载状态字典
    pretrain = torch.load(checkpoint_path)
    if 'model_state_dict' in pretrain.keys():
        state_dict = pretrain['model_state_dict']
    else:
        state_dict = pretrain['state_dict']

    # 如果模型是多GPU训练的，移除 'module.' 前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    # 加载模型权重
    model.load_state_dict(state_dict, strict=False)  # 允许不完全匹配

    # 如果需要在多GPU上运行模型
    if multi_gpu:
        # 使用DataParallel封装模型
        model = nn.DataParallel(model)

    return model



def evaluate_and_plot(model, test_loader, device, model_name, all_fprs=None, all_tprs=None, all_aucs=None, all_accuracies=None):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    correct_preds = 0
    total_preds = 0
    print(model_name)
    
    # 计算所有预测
    with torch.no_grad():
        for inx, (img, mask, label, clinical) in enumerate(test_loader):
            model.eval()
            img, label, clinical, mask = img.to(device), label.to(device), clinical.to(device), mask.to(device)

            if model_name == 'SegPrompt':
                cls = model(img, mask.float())
            elif model_name == 'TMSS':
                seg, cls = model([img, clinical])
            elif model_name == 'HyMNet':
                seg, cls = model(img, clinical)
            elif model_name == 'USCNet':
                seg, cls = model(img, clinical)
            elif model_name in ['ResNet34', 'ResNet50', 'ResNet101']:  # 处理 ResNet 模型
                cls = model(img)[-1]

            pred = torch.sigmoid(cls)
            pred = (pred > 0.5).float()
            all_preds.extend(pred.cpu().numpy())  # 类别预测
            all_labels.extend(label.cpu().numpy())  # 实际标签
            all_probs.extend(torch.sigmoid(cls).cpu().numpy())  # 预测的概率值

            # 计算准确率
            preds = (torch.sigmoid(cls) > 0.5).float()  # 二值化预测
            correct_preds += torch.sum(preds == label).item()
            total_preds += label.size(0)

    # 计算 AUC
    fpr, tpr, _ = roc_curve(all_labels, np.array(all_probs).flatten())
    auc_value = auc(fpr, tpr)

    # 计算准确率
    accuracy = correct_preds / total_preds

    # 将每个模型的 fpr, tpr 和 auc 保存到外部列表中
    if all_fprs is None:
        all_fprs = []
    if all_tprs is None:
        all_tprs = []
    if all_aucs is None:
        all_aucs = []
    if all_accuracies is None:
        all_accuracies = []

    all_fprs.append(fpr)
    all_tprs.append(tpr)
    all_aucs.append(auc_value)
    all_accuracies.append(accuracy)

    return all_fprs, all_tprs, all_aucs, all_accuracies, all_labels, all_preds

if __name__ == '__main__':
    # 数据集路径
    with open('configs/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    data_dir = dataset['data_dir']
    infos_name = dataset['infos_name']
    filter_volume = dataset['filter_volume']
    train_info, val_info = split_data(data_dir, infos_name, filter_volume, rate=0.8)
    train_dataloader = my_dataloader(data_dir, train_info, batch_size=1)
    test_dataloader = my_dataloader(data_dir, val_info, batch_size=1)

    # 定义模型实例和对应的权重文件名
    models_and_checkpoints = {
        "SegPrompt": {
            "model": SegPromptBackbone(in_channels=1, out_channels=1, img_size=48),
            "checkpoint": "SegPromp.pth"
        },
        "TMSS": {
            "model": DoubleFlow(in_channels=1, out_channels=1, img_size=(48, 48, 48), feature_size=16, patch_size=16),
            "checkpoint": "TMSS.pth"
        },
        "HyMNet": {
            "model": HyMNet(in_channels=1, out_channels=1, img_size=(48, 48, 48), feature_size=16, patch_size=16,
                            hidden_size=384, num_heads=12),
            "checkpoint": "HyMNet.pth"
        },
        "USCNet": {
            "model": KSCNet(in_channels=1, out_channels=1, img_size=(48, 48, 48), feature_size=16, patch_size=16,
                            hidden_size=384, num_heads=12),
            "checkpoint": "USCNet.pth"
        },
        "ResNet34": {
            "model": generate_model(model_depth=50, n_classes=2),  # 修改 ResNet34
            "checkpoint": "ResNet34.pth"
        },
        "ResNet50": {
            "model": generate_model(model_depth=34, n_classes=2),  # 修改 ResNet50
            "checkpoint": "ResNet50.pth"
        },
        "ResNet101": {
            "model": generate_model(model_depth=101, n_classes=2),  # 修改 ResNet101
            "checkpoint": "ResNet101.pth"
        }
    }

    # 模型权重所在目录
    checkpoint_dir = r'D:\zsq\KidneyStone\models_pth'

    # 检查是否可以使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化用于保存 AUC 曲线的列表
    all_fprs = []
    all_tprs = []
    all_aucs = []
    all_accuracies = []
    
    # 存储混淆矩阵数据
    confusion_matrix_data = {}

    # 遍历每个模型和对应的权重文件
    for model_name, info in models_and_checkpoints.items():
        model = info["model"]
        checkpoint_path = os.path.join(checkpoint_dir, info["checkpoint"])

        # 加载模型权重
        model = load_model(model, checkpoint_path, multi_gpu=False)
        model.to(device)

        # 打印模型信息
        print(f"Evaluating model: {model_name}")

        # 调用评估函数
        all_fprs, all_tprs, all_aucs, all_accuracies, all_labels, all_preds = evaluate_and_plot(
            model, test_dataloader, device, model_name, all_fprs, all_tprs, all_aucs, all_accuracies)
        
        # 保存混淆矩阵数据（排除ResNet系列）
        if model_name not in ['ResNet34', 'ResNet50', 'ResNet101']:
            confusion_matrix_data[model_name] = {
                'labels': all_labels,
                'preds': all_preds
            }

    # 绘制四个模型的混淆矩阵（2x2布局）
    plt.figure(figsize=(12, 10), dpi=200)
    for i, (model_name, data) in enumerate(confusion_matrix_data.items()):
        plt.subplot(2, 2, i+1)
        cm = confusion_matrix(data['labels'], (np.array(data['preds']) > 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-infectious', 'infectious'],
                    yticklabels=['Non-infectious', 'infectious'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name}', pad=20)  # 增加标题与图的间距
    plt.tight_layout()
    plt.savefig('combined_confusion_matrix.png')
    plt.show()

    plt.figure(figsize=(12, 10), dpi=300)
    for i, (model_name, data) in enumerate(confusion_matrix_data.items()):
        plt.subplot(2, 2, i + 1)
        cm = confusion_matrix(data['labels'], (np.array(data['preds']) > 0.5).astype(int))

        # 绘制热力图（去除所有标签，放大数字，正方形显示）
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='Blues',
                    cbar=False,  # 不显示颜色条
                    annot_kws={"size": 12},  # 进一步增大数字字体
                    xticklabels=False,  # 不显示x轴标签
                    yticklabels=False)  # 不显示y轴标签

        # 确保每个子图是正方形
        plt.gca().set_aspect('equal', adjustable='box')

        # 只保留模型名称标题
        plt.title(f'{model_name}', fontsize=16, pad=20)

        # 移除所有坐标轴标签
        plt.xlabel('')
        plt.ylabel('')

    plt.tight_layout()
    plt.savefig('combined_confusion_matrix.png', bbox_inches='tight')
    plt.show()

        # 调整坐标轴标签字体大小


    plt.tight_layout()
    plt.savefig('combined_confusion_matrix.png', bbox_inches='tight')  # 保存时确保完整显示
    plt.show()

    # ## 绘制所有模型的 AUC 曲线
    # plt.figure(dpi=300)
    # for i, model_name in enumerate(models_and_checkpoints.keys()):
    #     # 如果是 USCNet，加粗线条
    #     if model_name == 'USCNet':
    #         plt.plot(all_fprs[i], all_tprs[i], lw=4, label=f'{model_name} (AUC = {all_aucs[i]:.4f})', color='red')  # 加粗并设置为红色
    #     else:
    #         plt.plot(all_fprs[i], all_tprs[i], lw=2, label=f'{model_name} (AUC = {all_aucs[i]:.4f})')
    #
    # # 绘制对角线
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #
    # # 设置图形属性
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc='lower right')
    #
    # # 保存图像
    # plt.savefig('total_auc_curve.png', bbox_inches='tight')  # 保存总的 AUC 曲线
    # plt.show()
    #
    # # 打印每个模型的准确率
    # for model_name, accuracy in zip(models_and_checkpoints.keys(), all_accuracies):
    #     print(f'{model_name} Accuracy: {accuracy:.4f}')