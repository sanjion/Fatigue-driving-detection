import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from tqdm import tqdm
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
import numpy as np
from pathlib import Path
import os
from torchvision.models import ResNet50_Weights
import time
import detect
from utils.dataset import YOLOToFasterRCNNDataset,collate_fn
import logging
import logging.config  # Import the logging.config module
from utils.general import (
     set_logging,fps
)
from utils.metrics import fitness
from torch.cuda.amp import autocast, GradScaler
import val
from torchvision.models import ResNet18_Weights
from utils.plots import save_loss_plot,save_mAP,save_precise_plot,save_recall_plot
from torchinfo import summary
from thop import profile
import pandas as pd

# 定义骨干网络
# 加载预训练的骨干网络
def train(data,names ,save_dir,epoch_num,batch_size=16,backbone='resnet50',workers=8):
    LOGGING_NAME = "fasterrcnn"
    set_logging(LOGGING_NAME)  # run before defining LOGGER
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
    #backbone = resnet_fpn_backbone(backbone_name=backbone, weights=ResNet50_Weights.DEFAULT)
    # 使用 ResNet18 并加载其权重
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None)
    backbone.out_channels = 256  # 设置 FPN 输出通道数
    # 模型路径和类别数
    names = ["background"] + names  # 假设类别名称为 class_1, class_2, ...
    nc=len(names)
    # 加载模型
    model = FasterRCNN(backbone, num_classes=nc,min_size=400, max_size=600)  # 81 类（包括背景）
    # 将模型移动到 GPU
    model = model.cuda()
    # 生成输入张量
    input_tensor = torch.randn(1, 3, 640, 640).cuda()
    # 打印模型参数量
    summary(model, input_size=input_tensor.shape)

    best_fitness = 0.0
    w = save_dir + "/weights"  # weights dir
    Path(w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w + "/last.pt", w + "/best.pt"
    train_loss_list = []
    map_50_list = []
    precise_list = []
    recall_list = []
    map_list = []
    classifier_loss_list = []
    box_loss_list = []
    objectness_loss_list = []
    rpn_box_loss_list = []
    # 使用数据集和数据加载器
    train_dataset = YOLOToFasterRCNNDataset(data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
    # 训练模型
    for epoch in range(epoch_num):  # 假设训练 10 个 epoch
        LOGGER.info(("%10s"*3 + "%22s" * 4) % ("epoch","GPU_mem","TotalLoss", "ClassifierLoss", "BoxRegressionLoss", "ObjectnessLoss", "RPNBoxRegressionLoss"))
        progress_bar = tqdm(train_loader, bar_format=TQDM_BAR_FORMAT)  # 添加进度条
        scaler = GradScaler()
        model.train()
        batch_total_loss = 0
        loss_classifier = 0
        loss_box_reg = 0
        loss_objectness = 0
        loss_rpn_box_reg = 0
        for index,(images, targets) in  enumerate(progress_bar):
            images = [F.to_tensor(image).cuda() for image in images]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with autocast():
                # 前向传播
                loss_dict = model(images, targets)
                # losses = sum(loss for loss in loss_dict.values())
                # 获取各项损失
                loss_classifier +=loss_dict['loss_classifier'].item()
                loss_box_reg += loss_dict['loss_box_reg'].item()
                loss_objectness +=loss_dict['loss_objectness'].item()
                loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

                # 计算总损失
                total_loss = loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] + loss_dict['loss_objectness'] +loss_dict['loss_rpn_box_reg']
                total_classification_loss = loss_dict['loss_classifier']
                batch_total_loss = batch_total_loss +total_loss.item()
                # 缩放梯度并反向传播
            scaler.scale(total_loss).backward()
                # 更新模型参数
            scaler.step(optimizer)
            scaler.update()  # 更新 scaler，准备下一次训练
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            progress_bar.set_description("      %d/%d     %s  %.8f        %.8f          %.8f              %.8f            %.4f" % (epoch+1, epoch_num, mem,batch_total_loss / (index+1), loss_classifier / (index+1), loss_box_reg / (index+1), loss_objectness / (index+1),loss_rpn_box_reg / (index+1)))

        train_loss_list.append(batch_total_loss/len(progress_bar))
        classifier_loss_list.append(loss_classifier/len(progress_bar))
        box_loss_list.append(loss_box_reg/len(progress_bar))
        objectness_loss_list.append(loss_objectness/len(progress_bar))
        rpn_box_loss_list.append(loss_rpn_box_reg/len(progress_bar))
        
        #LOGGER.info()
        if epoch + 1 < epoch_num:
            results, maps, _ = val.run(data=data, batch_size=batch_size, model=model, task='train', names=names[1:],
                                       workers=workers)  # max dataloader workers (per RANK in DDP mode))
        else:
            results, maps, _ = val.run(data=data, batch_size=batch_size, save_dir=save_dir, model=model, task='val',
                                       names=names[1:],
                                       workers=workers)  # max dataloader workers (per RANK in DDP mode))
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
        'Objectness_Loss': objectness_loss_list,
        'RPN_Box_Loss': rpn_box_loss_list,
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
    test = os.path.join(data,"test","images")
    print("\r\n模型训练完成,开始推理:"+test)
    if not os.path.exists(test):
        test = os.path.join(data, "images", "test")
    if  os.path.exists(test):
        detect.run(input_folder=test,output_folder=os.path.join(save_dir,"detect"),names=names,model=model)


def create_folder(base_name):
    folder_name = base_name
    index = 1
    while os.path.exists(folder_name):
        folder_name = f"{base_name}{index}"
        index += 1

    os.makedirs(folder_name)
    return folder_name
if __name__ == "__main__":
    data = r'/home/ubuntu/faster_ssd/datas'

    names = ["Overlap","Suface pore","Undercut","Dent"] 
    batch_size = 8
    save_dir =create_folder(data+"/faster")
    workers=8
    epoch_num=100
    train(data=data, names=names,batch_size=batch_size,epoch_num=epoch_num,workers=workers,save_dir=save_dir,backbone="vgg16")