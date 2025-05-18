import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import os
from PIL import Image, ImageDraw, ImageFont
from thop import profile
import time
from torchinfo import summary
from tqdm import tqdm
# 推理函数
def inference(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).cuda()  # 将图像转换为 Tensor 并移动到 GPU

    # 执行推理
    with torch.no_grad():
        prediction = model(image_tensor)

    # 获取预测框、标签和得分
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    return image, boxes, labels, scores

class Colors:
    def __init__(self):
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17",
            "3DDB86", "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF",
            "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i):
        return self.palette[int(i) % self.n]

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

# 将目标框绘制到图像上
def draw_boxes(image, names, boxes, labels, scores, threshold=0.9):
    draw = ImageDraw.Draw(image)
    # 设置字体和大小
    font = ImageFont.truetype("arial.ttf", size=40)  # 字体文件和字体大小
    # 遍历所有预测框
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:  # 只绘制得分高于阈值的框
            xmin, ymin, xmax, ymax = map(int, box)
            color = colors(label.item())  # 获取对应类别的颜色
            text = f'{names[label.item()]}, {score.item():.2f}'

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            draw.text((xmin, ymin), text, font=font, fill=color)

    return image

def run(
    input_folder,
    names,
    model_path = None,
    output_folder=None,
    model=None,
    backbone_name='resnet18'):
    # 定义骨干网络（不加载预训练权重）
    if model is None:
        backbone = resnet_fpn_backbone(backbone_name=backbone_name, weights=None)
        backbone.out_channels = 256  # 设置 FPN 输出通道数
        names = ["background"] + names
        model = FasterRCNN(backbone, num_classes=len(names))  # 81 类（包括背景）

        # 加载自己训练好的模型权重
        model.load_state_dict(torch.load(model_path))  # 这里加载你的训练好的模型
    model.eval()  # 切换到评估模式
    # 将模型移到 GPU
    model = model.cuda()
    if output_folder == None:
       output_folder =os.path.join(os.path.dirname(input_folder),"detect")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.isdir(input_folder):
        # 获取所有图片路径
        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        image_paths = [os.path.join(input_folder, f) for f in image_files]
        for index, image_path in enumerate(tqdm(image_paths, desc="Processing Images")):
            # 调用推理函数进行测试
            image, boxes, labels, scores = inference(model, image_path)
            # 将预测框绘制到图像上
            image_with_boxes = draw_boxes(image, names, boxes, labels, scores)
            # 保存图像
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            image_with_boxes.save(output_path)
    else:
        # 调用推理函数进行测试
        image, boxes, labels, scores = inference(model, input_folder)
        # 将预测框绘制到图像上
        image_with_boxes = draw_boxes(image, names, boxes, labels, scores)
        # 显示图像
        plt.imshow(image_with_boxes)
        plt.axis('off')  # 不显示坐标轴
        plt.show()
        # 保存图像
        output_path = os.path.join(output_folder, os.path.basename(input_folder))
        image_with_boxes.save(output_path)

if __name__ == "__main__":
    data = r'D:\fasterrcnn7类\1'
    names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'train']
    #names = ['D00', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06','D07']
    backbone = 'resnet18'
    model_path = r"D:\fasterrcnn7类\last_model.pth"
    run(
        input_folder=data,
        model_path=model_path,
        names=names,
        model=None,
        backbone_name='resnet18')
