import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchinfo import summary

# 加载预训练的 Faster R-CNN 模型
model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# 定义输入形状
input_size = (1, 3, 800, 800)  # 批量大小为 1，3 通道，800x800 图像

# 打印模型详细信息
try:
    summary(model, input_size=input_size)
except Exception as e:
    print(f"Error occurred: {e}")