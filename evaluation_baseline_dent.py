#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from dent_faster_rcnn.imports import *
from imports import *
# from dent_faster_rcnn.cfg import *
from cfg import *
import pickle
# from dent_faster_rcnn.datasets.dent import *
from datasets.dent import *
# from dent_faster_rcnn.coco_eval import CocoEvaluator
from coco_eval import CocoEvaluator
import time
# from dent_faster_rcnn.coco_utils import get_coco_api_from_dataset
from coco_utils import get_coco_api_from_dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[ ]:


print("Loading files")

print("Evaluation on Dent Dataset")
with open("datalists/dent_images_path_list.txt", "rb") as fp:
    dent_image_path_list = pickle.load(fp)
with open("datalists/dent_anno_path_list.txt", "rb") as fp:
    dent_anno_path_list = pickle.load(fp)

train_img_paths = []
with open(os.path.join(dent_path, 'train.txt')) as f:
    train_img_paths = f.readlines()

for i in range(len(train_img_paths)):
    train_img_paths[i] = train_img_paths[i].strip('\n')
    train_img_paths[i] = train_img_paths[i] + '.jpg'
    img_dir = os.path.join(dent_path, 'images')
    train_img_paths[i] = os.path.join(img_dir, train_img_paths[i])

train_anno_paths = []
for i in range(len(train_img_paths)):
    train_anno_paths.append(train_img_paths[i].replace('images', 'Dent_annotations', 1))
    train_anno_paths[i] = train_anno_paths[i].replace('.jpg', '.xml')

train_img_paths, train_anno_paths = sorted(train_img_paths), sorted(train_anno_paths)

assert len(train_img_paths) == len(train_anno_paths)

new_train_img_paths, new_train_anno_paths = [], []
for i in range(len(train_img_paths)):
    if (train_img_paths[i] in dent_image_path_list):
        new_train_img_paths.append(train_img_paths[i])
        new_train_anno_paths.append(train_anno_paths[i])

assert len(new_train_img_paths) == len(new_train_anno_paths)

dataset_train = DENT(new_train_img_paths, new_train_anno_paths, None)
dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model.cuda()

model = get_model(len(dataset_train.classes))
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

checkpoint = torch.load('saved_models/' + model_name)
model.load_state_dict(checkpoint['model'])
print("Model Loaded successfully")

print("##### Dataloader is ready #######")

print("Getting coco api from dataset")
coco = get_coco_api_from_dataset(dl.dataset)
print("Done")

print("Evaluation in progress")
evaluate(model, dl, device=device)


# In[ ]:




