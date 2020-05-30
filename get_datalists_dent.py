# from dent_faster_rcnn.imports import *
from imports import *

# from dent_faster_rcnn.cfg import *

from cfg import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Creating datalist for Dent Dataset")
######################################################################################
root_anno_path = os.path.join(dent_path, 'Dent_annotations')
root_img_path = os.path.join(dent_path, 'images')

img_paths = [root_img_path]
anno_paths = [root_anno_path]

total_img_paths = []
for i in tqdm(range(len(img_paths))):
    img_names = os.listdir(img_paths[i])
    for j in range(len(img_names)):
        img_name = os.path.join(img_paths[i], img_names[j])
        total_img_paths.append(img_name)

total_anno_paths = []
for i in tqdm(range(len(anno_paths))):
    anno_names = os.listdir(anno_paths[i])
    for j in range(len(anno_names)):
        anno_name = os.path.join(anno_paths[i], anno_names[j])
        total_anno_paths.append(anno_name)

total_img_paths, total_anno_paths = sorted(total_img_paths), sorted(total_anno_paths)

###############################################################
def get_obj_bboxes(xml_obj):
    xml_obj = ET.parse(xml_obj)
    objects, bboxes = [], []

    for node in xml_obj.getroot().iter('object'):
        object_present = node.find('name').text

        xmin = max(int(node.find('bndbox/xmin').text), 0)
        xmax = max(int(node.find('bndbox/xmax').text), 0)
        ymin = max(int(node.find('bndbox/ymin').text), 0)
        ymax = max(int(node.find('bndbox/ymax').text), 0)
        objects.append(object_present)
        bboxes.append((xmin, ymin, xmax, ymax))
    return objects, bboxes

print("######### Checking ############")
print("Images without annotations found, fixing them")
cnt = 0

no_annotations = []
for i, a in tqdm(enumerate(total_anno_paths)):
    obj_anno_0 = get_obj_bboxes(total_anno_paths[i])
    if not obj_anno_0[0]:
        no_annotations.append(a)
        total_anno_paths.remove(a)
        a = a.replace('Dent_annotations', 'images')
        a = a.replace('xml', 'jpg')
        total_img_paths.remove(a)
        # print("Problematic", a)
        cnt += 1

for i in no_annotations:
    total_anno_paths.remove(i)
    i = i.replace('Dent_annotations', 'images')
    i = i.replace('xml', 'jpg')
    total_img_paths.remove(i)

print('Total number of images without annotations: ' + str(cnt))

print(len(total_anno_paths), len(total_img_paths))
assert len(total_anno_paths) == len(total_img_paths)

with open("datalists/dent_images_path_list.txt", "wb") as fp:
    pickle.dump(total_img_paths, fp)

with open("datalists/dent_anno_path_list.txt", "wb") as fp:
    pickle.dump(total_anno_paths, fp)

print("Saved successfully", "datalists/dent_images_path_list.txt")
print(len(total_anno_paths), len(total_img_paths))
