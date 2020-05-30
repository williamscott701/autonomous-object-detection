# from dent_faster_rcnn.imports import *
from imports import *
import pickle
# from dent_faster_rcnn.datasets.dent import *
from datasets.dent import *
# from dent_faster_rcnn.cfg import *
from cfg import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with open("datalists/dent_images_path_list.txt", "rb") as fp:
    img_paths = pickle.load(fp)
with open("datalists/dent_anno_path_list.txt", "rb") as fp:
    anno_paths = pickle.load(fp)

images = img_paths
annos = anno_paths

train_img_paths = []
with open(os.path.join(dent_path, 'train.txt')) as f:
    train_img_paths = f.readlines()

for i in range(len(train_img_paths)):
    train_img_paths[i] = train_img_paths[i].strip('\n')
    train_img_paths[i] = train_img_paths[i] + '.jpg'
    img_dir = os.path.join(dent_path , 'images')
    train_img_paths[i] = os.path.join(img_dir, train_img_paths[i])

train_anno_paths = []
for i in range(len(train_img_paths)):
    train_anno_paths.append(train_img_paths[i].replace('images', 'Dent_annotations',1))
    train_anno_paths[i] = train_anno_paths[i].replace('.jpg', '.xml')

train_img_paths, train_anno_paths = sorted(train_img_paths), sorted(train_anno_paths)

assert len(train_img_paths) == len(train_anno_paths)

new_train_img_paths, new_train_anno_paths = [], []

for i in range(len(train_img_paths)):
    if (train_img_paths[i] in images):
        new_train_img_paths.append(train_img_paths[i])
        new_train_anno_paths.append(train_anno_paths[i])

assert len(new_train_img_paths) == len(new_train_anno_paths)

dataset_train = DENT(new_train_img_paths, new_train_anno_paths, get_transform(train=True))
dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)

print("Loading done")

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                               num_classes)  # replace the pre-trained head with a new one
    return model.cuda()

print("Model initialization")
model = get_model(len(dataset_train.classes))
# torch.cuda.empty_cache()
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=6e-3)

try:
    os.mkdir('saved_models/')
except:
    pass

if ckpt:
    checkpoint = torch.load('saved_models/dent.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']


print('Training started')

for epoch in tqdm(range(num_epochs)):
    train_one_epoch(model, optimizer, dl, device, epoch, print_freq=200)
    lr_scheduler.step()

    save_name = 'saved_models/dent_' + str(epoch) + '.pth'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_name)
    print("Saved model", save_name)
