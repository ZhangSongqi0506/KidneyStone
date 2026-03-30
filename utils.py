import torch
import numpy as np
import os
import random
from skimage.transform import resize


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size

def calculate_acc_sigmoid(outputs, targets):
    batch_size = targets.size(0)
    pred = torch.round(outputs)
    correct = (pred == targets).float()
    n_correct_elems = correct.sum().item()
    return n_correct_elems / batch_size
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(12) + worker_id)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.isnan(val):
            val = 0
        if type(val) == torch.Tensor:
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter2(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def generate_patch_mask(img, pred_mask):
    oi_seg = torch.mul(img, pred_mask)
    zoom_seg = torch.cat([img, pred_mask], dim=1)
    return oi_seg, zoom_seg


def returnCAM(feature_conv, weight_softmax, idx, size=(256, 256, 256)):
    bz, nc, h, w, d = feature_conv.shape
    feature_conv = feature_conv.reshape((bz, nc, h * w * d))
    #zoom = Resize([256, 256, 128])
    cams = []
    for i in range(bz):
        cam_bi = torch.matmul(weight_softmax[idx[i], :], feature_conv[i, :, :])
        cam_bi = cam_bi.reshape((1, h, w, d))
        cam_img = (cam_bi - cam_bi.min()) / (cam_bi.max() - cam_bi.min())  # normalize
        cam_img = resize(cam_img.to('cpu').detach().numpy()[0], size)
        cam_img = torch.Tensor(cam_img)
        cam_img = cam_img.unsqueeze(0)
        cam_img = cam_img.unsqueeze(0)

        cams.append(cam_img)
    out = torch.cat(cams, dim=0)

    return out

def load_pretrain(path, model):
    if path:
        if os.path.exists(path):
            print("loading pretrained model from {}".format(path))
            pretrained_dict = torch.load(path)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print('pretrained model path not exists!')
    return model