import cv2
import numpy as np
import json
import os
import shutil
# 可视化coco格式的keypoint标注

# anno_path = "./coco/annotations/val_keypoint_527.json"
anno_path = "/disk3/wangsong01/Automated_Annotation_Project/data_kp.json"
# image_root = "./coco/val_imgs_keypoint/"
image_root = "/disk3/wangsong01/ai_sports/imgs"
save_root = "./keypoint_vis_output"
if os.path.exists(save_root):
    shutil.rmtree(save_root)
os.makedirs(save_root)

with open(anno_path, 'r') as f:
    data = json.load(f)

image_info = data["images"]
annos = data["annotations"]
categories = data["categories"][0]
skeleton = categories["skeleton"]
id2name_dict = {}
id2img_dict = {}
for image in image_info:
    id2name_dict[image["id"]] = image["file_name"]

for anno in annos:
    image_id = anno["image_id"]
    bbox = anno["bbox"]
    bbox = [int(x) for x in bbox]
    keypoints = anno["keypoints"]
    keypoints = [int(x) for x in keypoints]
    if image_id in id2img_dict:
        img = id2img_dict[image_id]
    else:
        image_name = id2name_dict[image_id]
        image_path = os.path.join(image_root, image_name)
        img = cv2.imread(image_path) 
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 255, 0), 2)
    for pos in range(0, len(keypoints), 3):
        cv2.circle(img, (keypoints[pos], keypoints[pos+1]), 3, (0, 255, 0), 3)
    for connection in skeleton:
        start_point = (keypoints[(connection[0]-1)*3], keypoints[(connection[0]-1)*3+1])
        end_point = (keypoints[(connection[1]-1)*3], keypoints[(connection[1]-1)*3+1])
        cv2.line(img, start_point, end_point, (0, 255, 255), 2)
    id2img_dict[image_id] = img

for key, img in id2img_dict.items():
    cv2.imwrite(os.path.join(save_root, str(key)+".jpg"), img)