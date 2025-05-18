import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from PIL import Image
import os
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from utils.dataset import YOLOToFasterRCNNDataset,collate_fn,ResizePad
import logging
import logging.config  # Import the logging.config module
from utils.general import (
     set_logging,fps
)
import time
import numpy as np
from pathlib import Path
from utils.metrics import fitness
from torch.cuda.amp import autocast, GradScaler
import detect
import val
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import ssd300_vgg16
from torchvision import transforms
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from utils.plots import save_loss_plot,save_mAP,save_precise_plot,save_recall_plot
from torchinfo import summary
from thop import profile
import pandas as pd

# 定义骨干网络
# 加载预训练的骨干网络
def train(data,names ,save_dir,epoch_num,batch_size=4,workers=8):
    LOGGING_NAME = "ssd"
    set_logging(LOGGING_NAME)  # run before defining LOGGER
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
    names = ["background"] + names  # 假设类别名称为 class_1, class_2, ...
    num_classes = len(names)
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

    # 将模型移动到 GPU
    model = model.cuda()
    # 生成输入张量
    input_tensor = torch.randn(1, 3, 640, 640).cuda()
    # 打印模型参数量
    summary(model, input_size=input_tensor.shape)
    #results = (0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    best_fitness = 0.0
    w = save_dir +"/weights"  # weights dir
    Path(w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w + "/last.pt", w + "/best.pt"

    # 使用数据集和数据加载器
    train_dataset = YOLOToFasterRCNNDataset(data,data="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
    train_loss_list = []
    map_50_list = []
    precise_list = []
    recall_list = []
    map_list = []
    classifier_loss_list = []
    box_loss_list = []
    # 训练模型
    for epoch in range(epoch_num):  # 假设训练 10 个 epoch
        LOGGER.info(("%10s"*3+"%22s" * 2) % ("epoch","GPU_mem","totalLoss", "bboxRegressionLoss", "classificationsLoss"))
        progress_bar = tqdm(train_loader, bar_format=TQDM_BAR_FORMAT)  # 添加进度条
        scaler = GradScaler()
        total_loss = 0
        bbox_regression = 0
        classification = 0
        model.train()
        for index,(images, targets) in enumerate(progress_bar):
            images = [F.to_tensor(image).cuda() for image in images]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with autocast():
                # 前向传播
                loss_dict = model(images, targets)
                # losses = sum(loss for loss in loss_dict.values())
                # 获取各项损失
                bbox_regression += loss_dict['bbox_regression'].item()
                classification += loss_dict['classification'].item()
                # 缩放梯度并反向传播
                total = loss_dict['bbox_regression']+loss_dict['classification']
            scaler.scale(total).backward()
                # 更新模型参数
            scaler.step(optimizer)
            scaler.update()  # 更新 scaler，准备下一次训练
            total_loss += total.item()
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            progress_bar.set_description("     %d/%d      %s   %.8f      %.8f          %.8f" % (epoch+1,epoch_num,mem,total_loss / (index+1), bbox_regression/ (index+1), classification/ (index+1)))

        train_loss_list.append(total_loss/len(progress_bar))
        classifier_loss_list.append(bbox_regression/len(progress_bar))
        box_loss_list.append(bbox_regression/len(progress_bar))
        
        
        
        if epoch+1<epoch_num:
            results, maps, _ = val.run(data=data, batch_size=batch_size, model=model, task='train', names=names[1:],workers=workers)  # max dataloader workers (per RANK in DDP mode))
        else:
            results, maps, _ = val.run(data=data, batch_size=batch_size, save_dir=save_dir, model=model, task='val', names=names[1:],workers=workers)  # max dataloader workers (per RANK in DDP mode))
        precise_list.append(results[0])
        recall_list.append(results[1])
        map_50_list.append(results[2])
        map_list.append(results[3])
            # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]

        if fi > best_fitness:
            best_fitness = fi
        torch.save(model.state_dict(), last)
        if best_fitness == fi:
            torch.save(model.state_dict(), best)
        LOGGER.info("\r")
            # Save loss plot.
            
        # 创建数据字典
    results_dict = {
        'Epoch': list(range(epoch_num)),
        'Total_Loss': train_loss_list,
        'Classifier_Loss': classifier_loss_list,
        'Box_Loss': box_loss_list,
        'Precision': precise_list,
        'Recall': recall_list,
        'mAP_50': map_50_list,
        'mAP': map_list
    }
    
    # 创建 DataFrame
    df = pd.DataFrame(results_dict)
    
    # 保存到 CSV 文件
    csv_path = os.path.join(save_dir, 'training_results.csv')
    df.to_csv(csv_path, index=False)
    LOGGER.info(f"Training metrics saved to {csv_path}")
    
    save_loss_plot(save_dir, train_loss_list)

    # Save mAP plot.
    save_mAP(save_dir, map_50_list, map_list)
    save_precise_plot(save_dir, precise_list)
    save_recall_plot(save_dir, recall_list)
    # callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)
    # print(*results,maps)
    LOGGER.info("Results saved to %s" % save_dir)
    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(input_tensor,))
    Gflops = flops / 1000000000
    print(f"GFLOPs: {Gflops:.2f}, Params: {params:.2f}")
    fps(model, input_tensor)
    time.sleep(1)
    test = os.path.join(data, "test", "images")
    print("\r\n模型训练完成,开始推理:" + test)
    if not os.path.exists(test):
        test = os.path.join(data, "images", "test")
    if os.path.exists(test):
        detect.run(input_folder=test, output_folder=os.path.join(save_dir, "detect"), names=names, model=model)



def create_folder(base_name):
    folder_name = base_name
    index = 1
    while os.path.exists(folder_name):
        folder_name = f"{base_name}{index}"
        index += 1

    os.makedirs(folder_name)
    return folder_name
if __name__ == "__main__":
    data = r'/home/ubuntu/faster_ssd/datasets/datasetes/'

    names = ["open eye","open mouth","closed eye","closed mouth"] 
    batch_size = 16
    save_dir =create_folder(data+"/ssd")
    workers=8
    epoch_num=100
    train(data=data, names=names,batch_size=batch_size,epoch_num=epoch_num,workers=workers,save_dir=save_dir)

