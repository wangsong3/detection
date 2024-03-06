import json
import os, cv2


#  可视化coco格式json标注中的box到图片上
import json
import shutil
import cv2

def select(json_path, outpath, image_path):
    json_file = open(json_path)
    infos = json.load(json_file)
    images = infos["images"]
    annos = infos["annotations"]
    assert len(images) == len(images)
    for i in range(len(images)):
        im_id = images[i]["id"]
        im_path = image_path + "/" + images[i]["file_name"]
        img = cv2.imread(im_path)
        for j in range(len(annos)):
            if annos[j]["image_id"] == im_id:
                x, y, w, h = annos[j]["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x2, y2 = x + w, y + h
                # object_name = annos[j][""]
                img = cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), thickness=2)
        img_name = outpath + "/" + images[i]["file_name"]
        cv2.imwrite(img_name, img)
                # continue
        print(i)

if __name__ == "__main__":

    # root = './riseHand_Dataset'
    # train_json = 'COCO/annotations/train2017.json'
    # train_path = 'COCO/train2017'
    # train_json = os.path.join(root, train_json)
    # train_path = os.path.join(root, train_path)
    train_json = "/disk3/wangsong01/Automated_Annotation_Project/data_det.json"
    train_path = "/disk3/wangsong01/ai_sports/imgs"
    visual_output = 'det_vis_output'
    if not os.path.exists(visual_output):
        os.makedirs(visual_output)

    select(train_json, visual_output, train_path)

