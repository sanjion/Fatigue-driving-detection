import torch

from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from torchvision.transforms import functional as F
import logging
import logging.config  # Import the logging.config module
import time
import matplotlib
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.dataset import YOLOToFasterRCNNDataset,collate_fn
from utils.general import (
    Profile,set_logging,
)
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection import ssd300_vgg16
matplotlib.use("Agg")  # for writing to files only
from torchvision.models import ResNet18_Weights
def process_batch(detections, labels, iouv):
    """
    Return a correct prediction matrix given detections and labels at various IoU thresholds.

    Args:
        detections (np.ndarray): Array of shape (N, 6) where each row corresponds to a detection with format
            [x1, y1, x2, y2, conf, class].
        labels (np.ndarray): Array of shape (M, 5) where each row corresponds to a ground truth label with format
            [class, x1, y1, x2, y2].
        iouv (np.ndarray): Array of IoU thresholds to evaluate at.

    Returns:
        correct (np.ndarray): A binary array of shape (N, len(iouv)) indicating whether each detection is a true positive
            for each IoU threshold. There are 10 IoU levels used in the evaluation.

    Example:
        ```python
        detections = np.array([[50, 50, 200, 200, 0.9, 1], [30, 30, 150, 150, 0.7, 0]])
        labels = np.array([[1, 50, 50, 200, 200]])
        iouv = np.linspace(0.5, 0.95, 10)
        correct = process_batch(detections, labels, iouv)
        ```

    Notes:
        - This function is used as part of the evaluation pipeline for object detection models.
        - IoU (Intersection over Union) is a common evaluation metric for object detection performance.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)
def run(
    data,
    names,
    save_dir = None,
    model_path=None,
    model=None,
    task="val",
    batch_size=32,
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    plots=True,
):
    LOGGING_NAME = "ssd"
    set_logging()  # run before defining LOGGER
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
    val_dataset = YOLOToFasterRCNNDataset(data, 'val')
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    names = ["background"] + names  # 假设类别名称为 class_1, class_2, ...
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    # 加载模型
    num_classes = len(names)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if task=="train":
        plots = False

    if model == None:
        size = 640
        # 加载预训练的 SSD 模型
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        # Retrieve the list of input channels.
        in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
        # List containing number of anchors based on aspect ratios.
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # The classification head.
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        # Image size for transforms.
        model.transform.min_size = (size,)
        model.transform.max_size = size

        # 加载自己训练好的模型权重
        model.load_state_dict(torch.load(save_dir + "/" + model_path))  # 这里加载你的训练好的模型
        model.to(device)
    confusion_matrix = ConfusionMatrix(nc=(num_classes - 1))
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    # 验证集评估
    model.eval()  # 切换到评估模式
    s = ("%10s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(device=device), Profile(device=device)  # profiling times
    #loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    single_cls = False
    TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
    with torch.no_grad():  # 禁用梯度计算
        progress_bar = tqdm(val_data_loader, desc=s, bar_format=TQDM_BAR_FORMAT)  # 添加进度条
        for batch_i, (im, targets) in enumerate(progress_bar):
            with dt[0]:
                images = [F.to_tensor(image).cuda() for image in im]
                tensortarget = torch.empty((0, 6))
                for i, target in enumerate(targets):
                    boxes = target['boxes']  # 边界框
                    labels = target['labels']  # 类别索引
                    # 沿着第二个轴（dim=1）合并
                    merged_tensor = torch.cat((labels.unsqueeze(1), boxes), dim=1)
                    # 定义要添加的整数（例如 0）
                    # integer_to_add = 0
                    # 创建一个列向量，形状为 (行数, 1)，填充为指定的整数
                    num_rows = merged_tensor.size(0)  # 获取行数
                    added_column = torch.full((num_rows, 1), i)
                    # 将整数列与合并后的张量拼接
                    final_tensor = torch.cat((added_column, merged_tensor), dim=1)
                    tensortarget = torch.cat((tensortarget, final_tensor), dim=0)

                targets = tensortarget
            with dt[1]:
                 preds = model(images)  # 输入模型进行推理
            # Metrics
            preds_temp = []
            for si, pred in enumerate(preds):
                # 解析预测结果
                boxes = pred['boxes']  # 边界框
                labels = pred['labels']  # 类别索引
                scores = pred['scores']  # 置信度

                # 将每个框的预测数据合并为 YOLOv5 格式 [x_min, y_min, x_max, y_max, conf, class_id]
                pred = torch.cat((boxes, scores.unsqueeze(1), labels.unsqueeze(1).float()),
                                 dim=1)  # (num_predictions, 6)
                preds_temp.append(pred)
                labels = targets[targets[:, 0] == si, 1:].to('cuda')
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions

                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                seen += 1
                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                        if plots:
                            confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                    continue
                # Predictions
                if single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()

                # Evaluate
                if nl:
                    # tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    tbox = labels[:, 1:5]  # target boxes
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Plot images
            # if plots and batch_i < 3:
            #    plot_images(im, targets, paths, f"val_batch{batch_i}_labels.jpg", names)  # labels
            #    plot_images(im, output_to_target(preds), paths,  f"val_batch{batch_i}_pred.jpg",names)  # pred

            # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=81)  # number of targets per class

        # Print results
        pf = "%10s" + "%11i" * 2 + "%11.3g" * 4  # print format
        LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
        # if nt.sum() == 0:
        #    LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

        # Print results per class
        # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        # Print results per class
        if task=="val" and num_classes > 1 and len(stats):
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        maps = np.zeros(num_classes) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        if task=="val":
            LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference per image " % t)

            # Plots
            if plots:
                confusion_matrix.plot(save_dir=save_dir, names=list(names.values())[1:])
            # join_threads(verbose=True)
            time.sleep(3)
        return (mp, mr, map50, map), maps, t
if __name__ == "__main__":
    
    data = r'/home/ubuntu/disk2/casual_code_/yolov8/VOCdevkit/'

    names = ['writing','reading','listening','turning_around', 'raising_head','standing','discussing','guiding'] 
    batch_size = 16
    save_dir = r"/home/ubuntu/disk2/casual_code_/yolov8/VOCdevkit/ssd/"
    model_path = r"/weights/best.pt"
    workers = 16
    run(data=data, batch_size=32, save_dir=save_dir, model_path=model_path, names=names,workers=workers)  # max dataloader workers (per RANK in DDP mode))


