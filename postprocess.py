import numpy as np
from utils import xywh2xyxy, get_iou

def RTMPose_decode(simcc_x, simcc_y, simcc_split_ratio=2.0):
        
    max_val_x = np.max(simcc_x, axis=2)
    x_locs = np.argmax(simcc_x, axis=2)
    max_val_y = np.max(simcc_y, axis=2)
    y_locs = np.argmax(simcc_y, axis=2)
    scores = np.maximum(max_val_x, max_val_y)
    keypoints = np.stack([x_locs, y_locs], axis=-1)
    keypoints = keypoints.astype(float) / simcc_split_ratio
    
    return keypoints, scores

def std_output(pred):
    """
    将 (1, 2100, 84)处理成(2100, 85)  85= box:4  conf:1 cls:80
    """
    pred = np.squeeze(pred)
    # pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    return pred  # (2100，85)

def nms(pred, conf_thres, iou_thres):
    """
    非极大值抑制nms
    Args:
        pred: 模型输出特征图
        conf_thres: 置信度阈值
        iou_thres: iou阈值
    Returns: 输出后的结果
    """
    box = pred[pred[..., 4] > conf_thres]  # 置信度筛选
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
 
    total_cls = list(set(cls))  # 记录图像内共出现几种物体
    output_box = []
    # 每个预测类别分开考虑
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        temp = box[:, :6]
        for j in range(len(cls)):
            # 记录[x,y,w,h,conf(最大类别概率),class]值
            if cls[j] == clss:
                temp[j][5] = clss
                cls_box.append(temp[j][:6])
        #  cls_box 里面是[x,y,w,h,conf(最大类别概率),class]
        cls_box = np.array(cls_box)
        sort_cls_box = sorted(cls_box, key=lambda x: -x[4])  # 将cls_box按置信度从大到小排序
        # box_conf_sort = np.argsort(-box_conf)
        # 得到置信度最大的预测框
        max_conf_box = sort_cls_box[0]
        output_box.append(max_conf_box)
        sort_cls_box = np.delete(sort_cls_box, 0, 0)
        # 对除max_conf_box外其他的框进行非极大值抑制
        while len(sort_cls_box) > 0:
            # 得到当前最大的框
            max_conf_box = output_box[-1]
            del_index = []
            for j in range(len(sort_cls_box)):
                current_box = sort_cls_box[j]
                iou = get_iou(max_conf_box, current_box)
                if iou > iou_thres:
                    # 筛选出与当前最大框Iou大于阈值的框的索引
                    del_index.append(j)
            # 删除这些索引
            sort_cls_box = np.delete(sort_cls_box, del_index, 0)
            if len(sort_cls_box) > 0:
                output_box.append(sort_cls_box[0])
                sort_cls_box = np.delete(sort_cls_box, 0, 0)
    return output_box

def cod_trf_det(result, pre, after, letterbox_image):
    """
    因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上
    Args:
        result:  [x,y,w,h,conf(最大类别概率),class]
        pre:    原尺寸图像
        after:  经过letterbox处理后的图像
    Returns: 坐标变换后的结果,并将xywh转换为左上角右下角坐标x1y1x2y2
    """
    res = np.array(result)
    x, y, w, h, conf, cls = res.transpose((1, 0))
    x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)  # 左上角点和右下角的点
    h_pre, w_pre, _ = pre.shape
    h_after, w_after, _ = after.shape
    if letterbox_image:
        scale = max(w_pre/w_after, h_pre/h_after)  # 缩放比例
        w_pre, h_pre = w_pre/scale, h_pre/scale  # 计算原图在等比例缩放后的尺寸
        x_move, y_move = abs(w_pre-w_after)//2, abs(h_pre-h_after)//2  # 计算平移的量
        ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
        ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
        ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
    else:
        w_scale, h_scale = w_pre/w_after, h_pre/h_after
        w_pre, h_pre = w_pre/w_scale, h_pre/h_scale
        x_move, y_move = abs(w_pre-w_after)//2, abs(h_pre-h_after)//2  # 计算平移的量
        ret_x1, ret_x2 = (x1 - x_move) * w_scale, (x2 - x_move) * w_scale
        ret_y1, ret_y2 = (y1 - y_move) * h_scale, (y2 - y_move) * h_scale
        ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
    return ret  # x1y1x2y2

def cod_trf_pose(keypoints, pre, after, bbox):
    """
    因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上
    Args:
        keypoints:  []
        pre:    原尺寸图像
        after:  经过letterbox处理后的图像
    Returns: 坐标变换后的结果,并将xywh转换为左上角右下角坐标x1y1x2y2
    """
    h_pre, w_pre, _ = pre.shape
    h_after, w_after, _ = after.shape
    x1, y1, x2, y2 = bbox

    w_scale, h_scale = w_pre/w_after, h_pre/h_after

    for i, keypoint in enumerate(keypoints):
        keypoints[i] = [keypoint[0] * w_scale, keypoint[1] * h_scale]
        keypoint[0] += x1
        keypoint[1] += y1

    return keypoints  # x1y1x2y2