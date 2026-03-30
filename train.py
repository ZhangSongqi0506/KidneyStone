import logging
import time
import os
import argparse
import json
import re
import torch.nn.functional as F
import monai.losses
from monai.bundle import ConfigParser
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys
from tqdm import tqdm
from src.models.networks.module import ResEncoder, CAL_Net
from src.models.networks.resnet import generate_model
from src.models.networks.sc_net import SC_Net
import shutil
from utils import AverageMeter, generate_patch_mask, returnCAM, load_pretrain
from src.dataloader.load_data import split_data, my_dataloader
from torch.nn import DataParallel
import itertools
import functools
import sklearn.metrics
from sklearn.metrics import accuracy_score

def main(args, save_path):
    config = ConfigParser()
    config.read_config(args.config_file)
    task = [int(i) for i in re.findall('\d', str(args.task))]
    print(task)
    train_seg = True if 0 in task else False
    train_cla = True if 1 in task else False

    # save_path = os.path.join(args.output_path, "seg-{}_cla-{}".format(train_seg, train_cla))
    # start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # save_path = os.path.join(save_path, start_time)

    model_dir = os.path.join(save_path, "models")
    summary_dir = os.path.join(save_path, "summarys")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(summary_dir, 'train'), flush_secs=2)
    test_writer = SummaryWriter(os.path.join(summary_dir, 'test'), flush_secs=2)

    #model
    [d, h, w] = [int(i) for i in re.findall('\d+',args.input_size)]
    img_size = (d//16, h//16, w//16)

    sc_net = SC_Net(in_channels=512, out_features=args.num_classes, img_size=img_size, cla=train_cla, seg=train_seg)
    sc_net = load_pretrain(args.pretrain_sc, sc_net)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        sc_net = DataParallel(sc_net)

    sc_net.to(device)
    optimizer = torch.optim.Adam(
        params=itertools.chain(sc_net.parameters()),
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=args.milestones,
        gamma=args.gamma
    )

    # loss
    criterion_seg = monai.losses.DiceLoss()
    criterion_cla = torch.nn.CrossEntropyLoss()
    criterion_weight = [float(i) for i in eval(args.loss_weights)]
    metrics_seg_list = config.get_parsed_content("VALIDATION#metrics#seg")
    metrics_seg = {k.func.__name__: k for k in metrics_seg_list if type(k) == functools.partial}
    metrics_seg.update({k.__class__.__name__: k for k in metrics_seg_list if type(k) != functools.partial})
    metrics_cla_list = config.get_parsed_content("VALIDATION#metrics#cla")
    metrics_cla = {}
    for k in metrics_cla_list:
        if type(k) == functools.partial:  # partial func
            metrics_cla[k.func.__name__] = k
        elif type(k) == str:  # func
            name = k.split('.')[-1]
            metrics_cla[name] = eval(k)
        else:
            metrics_cla[k.__class__.__name__] = k
    # data
    with open('configs/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    data_dir = dataset['data_dir']
    infos_name = dataset['infos_name']
    filter_volume = dataset['filter_volume']
    train_info, val_info = split_data(data_dir, infos_name, filter_volume, rate=0.8)
    train_data_loader = my_dataloader(data_dir,
                                      train_info,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers)
    test_data_loader = my_dataloader(data_dir,
                                     val_info,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers)

    print("Start training")
    running_loss = AverageMeter()
    running_loss_seg = AverageMeter()
    running_loss_cla = AverageMeter()
    metrics_seg_values = {k: AverageMeter() for k in metrics_seg}
    best_metric = {k: {"epoch": 0, "value": 0} for k in config["VALIDATION"]["save_best_metric"]}

    for epoch in tqdm(range(args.epochs)):
        s_t = time.time()
        sc_net.train()
        running_loss.reset()
        running_loss_seg.reset()
        running_loss_cla.reset()
        for k in metrics_seg_values:
            metrics_seg_values[k].reset()

        gt = []
        cla_pred = []
        for batch_idx, (img, seg_label, cla_label) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            bs, c, d, w, h = img.shape
            img = img.to(device)
            seg_label = seg_label.to(device)
            seg_loss = 0
            cla_loss = 0
            cla_label = cla_label.long()[:, 0].to(device)
            cla_out, seg_out = sc_net(img)
            if train_seg:
                pred_mask = torch.where(seg_out > 0.5, 1, 0).byte()
                seg_loss += criterion_seg(seg_out, seg_label)
                for k in metrics_seg:
                    res = metrics_seg[k](pred_mask, seg_label)
                    if type(res) == torch.Tensor and res.shape[0] > 0:
                        res = torch.mean(res[~torch.isnan(res)])
                    metrics_seg_values[k].update(res, bs)

            if train_cla:
                cla_loss += criterion_cla(cla_out, cla_label)
                gt.append(cla_label.cpu())
                cla_pred.append(F.softmax(cla_out, dim=-1).argmax(1, keepdim=True).cpu())

            w_s, w_c = criterion_weight
            loss = w_s * seg_loss + w_c * cla_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.update(loss, bs)
            if train_seg:
                running_loss_seg.update(seg_loss, bs)
            if train_cla:
                running_loss_cla.update(cla_loss, bs)

        if train_cla:
            gt = torch.cat(gt, dim=0)
            cla_pred = torch.cat(cla_pred, dim=0)
            metrics_cla_values = {k: m(gt, cla_pred) for k, m in metrics_cla.items()}
        else:
            metrics_cla_values = {k: 0 for k in metrics_cla}

        scheduler.step()
        print("----------------------------------------------------------------------------------")
        print("Epoch [%d/%d], time: %.2f" % (epoch, args.epochs, time.time() - s_t))
        print("Loss: total-{}_segmentation-{}_classification-{}.".format(
            running_loss.avg, running_loss_seg.avg, running_loss_cla.avg
        ))
        print("Metrics:", {k: metrics_seg_values[k].avg for k in metrics_seg_values}, metrics_cla_values)

        train_writer.add_scalar("Loss/total", running_loss.avg, epoch)
        train_writer.add_scalar("Loss/segmentation", running_loss_seg.avg, epoch)
        if train_cla:
            train_writer.add_scalar("Loss/classification", running_loss_cla.avg, epoch)
        train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        for k in metrics_seg_values:
            train_writer.add_scalar("Metrics_seg/" + k, metrics_seg_values[k].avg, epoch)
        if train_cla:
            for k in metrics_cla_values:
                train_writer.add_scalar("Metrics_cla/" + k, metrics_cla_values[k], epoch)
        if (epoch+1) % args.save_epoch == 0:
            torch.save(sc_net.state_dict(), os.path.join(model_dir, "sc_net_params_epo-{}.pkl".format(epoch)))

        if (epoch+1) % 5 == 0:
            print("Start testing...")
            s_t = time.time()
            sc_net.eval()
            running_loss.reset()
            running_loss_seg.reset()
            running_loss_cla.reset()
            for k in metrics_seg_values:
                metrics_seg_values[k].reset()

            gt = []
            cla_pred = []
            with torch.no_grad():

                for batch_idx, (img, seg_label, cla_label) in tqdm(enumerate(test_data_loader),
                                                                   total=len(test_data_loader)):
                    bs, c, d, w, h = img.shape
                    img = img.to(device)
                    seg_label = seg_label.to(device)
                    seg_loss = 0
                    cla_loss = 0
                    cla_label = cla_label.long()[:, 0].to(device)
                    cla_out, seg_out = sc_net(img)
                    if train_seg:
                        pred_mask = torch.where(seg_out > 0.5, 1, 0).byte()
                        seg_loss += criterion_seg(seg_out, seg_label)
                        for k in metrics_seg:
                            res = metrics_seg[k](pred_mask, seg_label)
                            if type(res) == torch.Tensor and res.shape[0] > 0:
                                res = torch.mean(res[~torch.isnan(res)])
                            metrics_seg_values[k].update(res, bs)

                    if train_cla:
                        cla_loss += criterion_cla(cla_out, cla_label)
                        gt.append(cla_label.cpu())
                        cla_pred.append(F.softmax(cla_out, dim=-1).argmax(1, keepdim=True).cpu())

                    w_s, w_c = criterion_weight
                    loss = w_s * seg_loss + w_c * cla_loss
                    running_loss.update(loss, bs)
                    if train_seg:
                        running_loss_seg.update(seg_loss, bs)
                    if train_cla:
                        running_loss_cla.update(cla_loss, bs)

                if train_cla:
                    gt = torch.cat(gt, dim=0)
                    cla_pred = torch.cat(cla_pred, dim=0)
                    metrics_cla_values = {k: m(gt, cla_pred) for k, m in metrics_cla.items()}
                else:
                    metrics_cla_values = {k: 0 for k in metrics_cla}

                print("Test results:")
                print("Loss: total-{}_segmentation-{}_classification-{}.".format(
                    running_loss.avg, running_loss_seg.avg, running_loss_cla.avg
                ))
                print("Metrics:", {k: metrics_seg_values[k].avg for k in metrics_seg_values}, metrics_cla_values)
                test_writer.add_scalar("Loss/total", running_loss.avg, epoch)
                test_writer.add_scalar("Loss/segmentation", running_loss_seg.avg, epoch)
                test_writer.add_scalar("Loss/classification", running_loss_cla.avg, epoch)

                for k in metrics_seg_values:
                    test_writer.add_scalar("Metrics_seg/" + k, metrics_seg_values[k].avg, epoch)
                for k in metrics_cla_values:
                    test_writer.add_scalar("Metrics_cla/" + k, metrics_cla_values[k], epoch)

                for k in best_metric:
                    if k in metrics_seg_values and metrics_seg_values[k].avg > best_metric[k]["value"]:
                        print('{} increased ({:.6f} --> {:.6f}).'.format(k, best_metric[k]["value"],
                                                                         metrics_seg_values[k].avg))
                        torch.save(sc_net.state_dict(), os.path.join(model_dir, "max_{}-sc_net.pkl".format(k)))
                        best_metric[k]["value"] = metrics_seg_values[k].avg
                        best_metric[k]["epoch"] = epoch
                    if k in metrics_cla_values and metrics_cla_values[k] > best_metric[k]["value"]:
                        print('{} increased ({:.6f} --> {:.6f}).'.format(k, best_metric[k]["value"],
                                                                         metrics_cla_values[k]))

                        torch.save(sc_net.state_dict(), os.path.join(model_dir, "max_{}-sc_net.pkl".format(k)))
                        best_metric[k]["value"] = metrics_cla_values[k]
                        best_metric[k]["epoch"] = epoch

            with open(os.path.join(model_dir, "best_metrics.json"), "w") as f:
                json.dump(best_metric, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default='configs/config.yaml')
    parser.add_argument('--task', type=str, default=[0, 1])
    parser.add_argument('--pretrain-sc', type=str, default=None)
    parser.add_argument('--input-path', type=str, default='/home/KidneyData/data')
    parser.add_argument('--output-path', type=str, default='./results')
    parser.add_argument('--input-size', type=str, default="48, 48, 48")
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--weight-decay', type=float, default=0.00001)
    parser.add_argument('--milestones', type=list, default=[100])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--loss-weights', type=str, default=[0.3, 0.7])
    parser.add_argument('--save-dir', type=str, default='./Save')
    parser.add_argument('--num-workers', type=int, default=0)

    opt = parser.parse_args()
    args_dict = vars(opt)

    # 将参数字典保存为 JSON 文件
    now = time.strftime('%y%m%d%H%M', time.localtime())
    if not os.path.exists(f'./results/{now}'):
        os.makedirs(f'./results/{now}')
    save_path = f'./results/{now}'
    with open(os.path.join(save_path, 'train_config.json'), 'w') as fp:
        json.dump(args_dict, fp, indent=4)
    print(f"Training configuration saved to {now}")

    # with open(f'./configs/training_config_{now}.json', 'w') as fp:
    #     json.dump(args_dict, fp, indent=4)
    #
    # print(f"Training configuration saved to training_config_{now}.json")

    main(opt, save_path=save_path)