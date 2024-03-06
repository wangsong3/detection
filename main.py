import os
import cv2
import numpy as np
import onnxruntime as rt
import json
from preprocess import resize_image, img2input
from onnx_inference import onnx_infer_det, onnx_infer_pose
from postprocess import std_output, nms, cod_trf_det, RTMPose_decode, cod_trf_pose
from utils import expand_bbox, get_padding_bbox, draw

def create_instance_coco(type="keypoint"):
    if type not in ["keypoint", "detection"]:
        raise ValueError("Type must be 'keypoint' or 'detection'")
    instance = {}
    if type == "keypoint":
        instance['categories'] = [{"supercategory": "person","id": 1, "name": "person", "keypoints": ["nose", "left_eye", "right_eye","left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", 
                    "left_ankle", "right_ankle"], "skeleton":[[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}]
    elif type == "detection":
        instance['categories'] = {"supercategory": "none","id": 1,"name": "person"}
    instance['images'] = []
    instance['annotations'] = []
    return instance

if __name__ == "__main__":
    det_onnx_path = "/disk3/wangsong01/yolov8/yolov8l.onnx"
    pose_onnx_path = "/disk3/wangsong01/yolov8/rtmpose_l.onnx"
    img_root = "/disk3/wangsong01/ai_sports/online_imgs"
    save_root = "./infer_results/"

    i = 0

    imgs = os.listdir(img_root)

    instance_kp = create_instance_coco("keypoint")
    instance_det = create_instance_coco("detection")

    anno_id = 0
    for i, img_name in enumerate(imgs):
        print(f"正在计算第{i}帧图片...")
        img_path = os.path.join(img_root, img_name)
        ori_img = cv2.imread(img_path)
        img = resize_image(ori_img, (640,640), True)
        input_img = img2input(img)
        pred = onnx_infer_det(input_img, det_onnx_path)[0]
        pred = std_output(pred)
        output_res = nms(pred, 0.8, 0.5)
        # 只保留 person 的检测结果
        output_res = [res for res in output_res if int(res[5])==0]

        output_res = cod_trf_det(output_res, ori_img, img, True)
        res_img = ori_img
        keypoints_per_img = []

        instance_kp['images'].append({"license":1, "file_name": img_name, "coco_url":"", "height": ori_img.shape[0], "width":ori_img.shape[1], "data_captured":"", "flickr_url":"", "id":i})
        instance_det['images'].append({"license":1, "file_name": img_name, "coco_url":"", "height": ori_img.shape[0], "width":ori_img.shape[1], "data_captured":"", "flickr_url":"", "id":i})

        for j, res in enumerate(output_res):
            x1, y1, x2, y2, conf, cls = res
            bbox_det = [x1,y1, x2-x1, y2-y1]
            bbox_det = [float(v) for v in bbox_det]
            instance_det['annotations'].append({"id":anno_id, "image_id":i, "category_id":1, "segmentation":[], "area":bbox_det[2]*bbox_det[3],"bbox":bbox_det,"iscrowd": 0, "ignore": 0})
            x1 ,y1, x2, y2 = expand_bbox([x1, y1, x2, y2])
            x1, y1, x2, y2 = get_padding_bbox([x1, y1, x2 ,y2])
            bbox_kp = [x1,y1, x2-x1, y2-y1]
            output_res[j][:4] = x1, y1, x2, y2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            img_before = ori_img[y1:y2, x1:x2,:]
            img = resize_image(img_before, (256, 192), False)
            input_img = img2input(img)
            simcc_x, simcc_y = onnx_infer_pose(input_img, pose_onnx_path)
            keypoints, scores = RTMPose_decode(simcc_x, simcc_y)
            keypoints, scores = keypoints[0], scores[0]
            keypoints = cod_trf_pose(keypoints, img_before, img, [x1, y1, x2, y2])
            
            visable = np.ones((17, 1), dtype=int) * 2
            keypoints_coco = np.concatenate((keypoints, visable), axis=1, dtype=object)

            keypoints = keypoints.flatten()
            keypoints_coco = keypoints_coco.flatten()

            instance_kp['annotations'].append({"id":anno_id, "image_id":i, "category_id":1, "segmentation":[[]], "area":bbox_kp[2]*bbox_kp[3],"bbox":bbox_kp,"iscrowd": 0,"keypoints":keypoints_coco.tolist(),"num_keypoints": 17})
            anno_id += 1
            # keypoints_per_img.append(keypoints)


            # keypoints = [int(x) for x in keypoints]
            if i<10:
                keypoints = [int(v) for v in keypoints]
                for pos in range(0, len(keypoints), 2):
                    cv2.circle(res_img, (keypoints[pos], keypoints[pos+1]), 3, (0, 255, 0), 3)

        if i<10:
            res_img = draw(output_res, res_img)
            cv2.imwrite(os.path.join(save_root, img_name), res_img)

    with open('data_kp.json', 'w') as f:
        json.dump(instance_kp, f, indent=4)
    with open('data_det.json', 'w') as f:
        json.dump(instance_det, f, indent=4)
    