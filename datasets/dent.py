import torch, os
from pathlib import Path
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from torch import Tensor, FloatTensor
import utils
# import dent_faster_rcnn.transforms as T
import transforms as T
# from dent_faster_rcnn.coco_eval import CocoEvaluator
from coco_eval import CocoEvaluator
import time
# from dent_faster_rcnn.coco_utils import get_coco_api_from_dataset
from coco_utils import get_coco_api_from_dataset

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    # transforms.append(T.Normalize(mean=(0.3520, 0.3520, 0.3520),std=(0.2930, 0.2930, 0.2930)))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class DENT(torch.utils.data.Dataset):
    def __init__(self, list_img_path, list_anno_path, transforms=None):
        super(DENT, self).__init__()
        self.img = list_img_path
        self.anno = list_anno_path
        self.transforms = transforms
        self.classes = {'Unknown':0, 'Dent': 1}

    def __len__(self):
        return len(self.img)

    def get_height_and_width(self, idx):
        img_path = os.path.join(img_path, self.img[idx])
        img = Image.open(img_path).convert("RGB")
        dim_tensor = torchvision.transforms.ToTensor()(img).shape
        height, width = dim_tensor[1], dim_tensor[2]
        return height, width

    def get_label_bboxes(self, xml_obj):
        xml_obj = ET.parse(xml_obj)
        objects, bboxes = [], []

        for node in xml_obj.getroot().iter('object'):
            object_present = node.find('name').text
            xmin = max(int(node.find('bndbox/xmin').text), 0)
            xmax = max(int(node.find('bndbox/xmax').text), 0)
            ymin = max(int(node.find('bndbox/ymin').text), 0)
            ymax = max(int(node.find('bndbox/ymax').text), 0)

            objects.append(self.classes[object_present])
            bboxes.append((xmin, ymin, xmax, ymax))
        return Tensor(objects), Tensor(bboxes)

    def __getitem__(self, idx):
        img_path = self.img[idx]
        img = Image.open(img_path).convert("RGB")

        labels = self.get_label_bboxes(self.anno[idx])[0]
        bboxes = self.get_label_bboxes(self.anno[idx])[1]
        labels = labels.type(torch.int64)
        img_id = Tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros(len(bboxes, ), dtype=torch.int64)

        ##
#         keep = (bboxes[:, 3] > bboxes[:, 1]) & (bboxes[:, 2] > bboxes[:, 0])
#         print(all(keep))
#         bboxes = bboxes[keep]
#         labels = labels[keep]
#         area = area[keep]
#         iscrowd = iscrowd[keep]
        ##

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
