import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection import ssd300_vgg16
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
def draw_boxes(image, names, boxes, labels, scores, threshold=0.5):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size=20)
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            xmin, ymin, xmax, ymax = map(int, box)
            color = colors(label.item())
            text = f'{names[label.item()]}, {score.item():.2f}'
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            draw.text((xmin, ymin), text, font=font, fill=color)
    return image


def run(input_folder, names, model_path=None, output_folder=None, model=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if model is None:
        names = ["background"] + names
        num_classes=len(names)
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
    model.eval().cuda()
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(input_folder), "detect")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)



    if os.path.isdir(input_folder):
        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        image_paths = [os.path.join(input_folder, f) for f in image_files]
        num =len(image_paths)

        for index, image_path in enumerate(tqdm(image_paths, desc="Processing Images")):
            image, boxes, labels, scores = inference(model, image_path)
            image_with_boxes = draw_boxes(image, names, boxes, labels, scores)
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            image_with_boxes.save(output_path)
    else:
        image, boxes, labels, scores = inference(model, input_folder)
        image_with_boxes = draw_boxes(image, names, boxes, labels, scores)
        plt.imshow(image_with_boxes)
        plt.axis('off')
        plt.show()
        output_path = os.path.join(output_folder, os.path.basename(input_folder))
        image_with_boxes.save(output_path)


if __name__ == "__main__":
    data = r'/home/ubuntu/disk2/casual_code_/ssd_faster/pre/datasets/images/test'
    # names = ["Bean", "Bottle_Gourd", "Brinjal", "Broccoli", "Cabbage", "Cauliflower", "Capsicum", "Carrot", "Papaya",
    #         "Potato", "Pumpkin", "Radish", "Pumpkin", "Radish"]
    names = ['loquat']
    batch_size = 16
    save_dir = r"/home/ubuntu/disk2/casual_code_/ssd_faster/pre/datasets/ssd1/detect"
    model_path = r"../weights/best.pt"
    run(input_folder=data, model_path=model_path, names=names)
