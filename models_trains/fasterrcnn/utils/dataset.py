import torch
from torch.utils.data import DataLoader
from PIL import Image
import os
# 自定义数据集类，将 YOLO 标签格式转为 Faster R-CNN 所需格式
IMG_FORMATS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm") # include image suffixes
class YOLOToFasterRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, path,data="train",transform=None):
        self.path = path
        self.images_dir = os.path.join(path,"images",data)
        self.labels_dir = os.path.join(path,"labels",data)
        if not os.path.exists(self.images_dir):
            self.images_dir = os.path.join(path, data, "images")
            self.labels_dir = os.path.join(path, data, "labels")
        self.transform = transform
        self.image_files = []
        #extensions = ('jpg', 'jpeg', 'png')
        for filename in os.listdir(self.images_dir):
            # 获取文件扩展名，返回的是一个元组 (文件名, 扩展名)
            _, ext = os.path.splitext(filename)
            # 去掉扩展名前面的 "." 并转换为小写
            ext = ext.lstrip('.').lower()
            # 检查是否为文件且扩展名符合要求
            if os.path.isfile(os.path.join(self.images_dir, filename)) and ext in IMG_FORMATS and os.path.isfile(os.path.join(self.labels_dir, filename.rsplit('.', 1)[0]+".txt")):
                self.image_files.append(filename)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx].rsplit('.', 1)[0]+".txt")  # 假设图片是 .jpg 格式

        image = Image.open(image_path).convert("RGB")
        boxes = []
        labels = []
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])+1
                x_center, y_center, width, height = map(float, parts[1:])

                # 将 YOLO 格式的标签转为 Faster R-CNN 需要的 [xmin, ymin, xmax, ymax]
                xmin = (x_center - width / 2) * image.width
                ymin = (y_center - height / 2) * image.height
                xmax = (x_center + width / 2) * image.width
                ymax = (y_center + height / 2) * image.height

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)
        if len(boxes)==0:
            boxes = torch.empty((0, 4))  # 目标框为空
            labels = torch.empty((0), dtype=torch.int64)  # 标签为空
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))