import cv2

cls_name = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def expand_bbox(box, ratio=6/5):
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    new_w, new_h = w*ratio, h*ratio
    new_x1 = x1 + w/2 - new_w/2
    new_x2 = x2 - w/2 + new_w/2
    new_y1 = y1 + h/2 - new_h/2
    new_y2 = y2 - h/2 + new_h/2
    if new_x1<0:
        new_x1, new_x2 = 0, new_x2-new_x1
    if new_y1<0:
        new_y1, new_y2 = 0, new_y2-new_y1
    if new_x2>1920:
        new_x1, new_x2 = new_x1+(1920-new_x2), 1920
    if new_y2>1080:
        new_y1, new_y2 = new_y1+(1080-new_y2), 1080
    return new_x1, new_y1, new_x2, new_y2

def get_padding_bbox(box, ratio=3/4):
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    if w/h > ratio:
        new_h = w/ratio
        new_w = w
    else:
        new_h = h
        new_w = h*ratio
    new_x1 = x1 + w/2 - new_w/2
    new_x2 = x2 - w/2 + new_w/2
    new_y1 = y1 + h/2 - new_h/2
    new_y2 = y2 - h/2 + new_h/2
    if new_x1<0:
        new_x1, new_x2 = 0, new_x2-new_x1
    if new_y1<0:
        new_y1, new_y2 = 0, new_y2-new_y1
    if new_x2>1920:
        new_x1, new_x2 = new_x1+(1920-new_x2), 1920
    if new_y2>1080:
        new_y1, new_y2 = new_y1+(1080-new_y2), 1080
    return new_x1, new_y1, new_x2, new_y2

def xywh2xyxy(*box):
    """
    将xywh转换为左上角点和左下角点
    Args:
        box:
    Returns: x1y1x2y2
    """
    ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
          box[0] + box[2] // 2, box[1] + box[3] // 2]
    return ret

def get_inter(box1, box2):
    """
    计算相交部分面积
    Args:
        box1: 第一个框
        box2: 第二个框
    Returns: 相交部分的面积
    """
    x1, y1, x2, y2 = xywh2xyxy(*box1)
    x3, y3, x4, y4 = xywh2xyxy(*box2)
    # 验证是否存在交集
    if x1 >= x4 or x2 <= x3:
        return 0
    if y1 >= y4 or y2 <= y3:
        return 0
    # 将x1,x2,x3,x4排序，因为已经验证了两个框相交，所以x3-x2就是交集的宽
    x_list = sorted([x1, x2, x3, x4])
    x_inter = x_list[2] - x_list[1]
    # 将y1,y2,y3,y4排序，因为已经验证了两个框相交，所以y3-y2就是交集的宽
    y_list = sorted([y1, y2, y3, y4])
    y_inter = y_list[2] - y_list[1]
    # 计算交集的面积
    inter = x_inter * y_inter
    return inter
 
def get_iou(box1, box2):
    """
    计算交并比： (A n B)/(A + B - A n B)
    Args:
        box1: 第一个框
        box2: 第二个框
    Returns:  # 返回交并比的值
    """
    box1_area = box1[2] * box1[3]  # 计算第一个框的面积
    box2_area = box2[2] * box2[3]  # 计算第二个框的面积
    inter_area = get_inter(box1, box2)
    union = box1_area + box2_area - inter_area   #(A n B)/(A + B - A n B)
    iou = inter_area / union
    return iou

def draw(res, image):
    """
    将预测框绘制在image上
    Args:
        res: 预测框数据
        image: 原图
        cls: 类别列表，类似["apple", "banana", "people"]  可以自己设计或者通过数据集的yaml文件获取
    Returns:
    """
    for r in res:
        # 画框
        image = cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 0, 255), 2)
        # 表明类别
        text = "{}:{}".format(cls_name[int(r[5])], \
                               round(float(r[4]), 2))
        h, w = int(r[3]) - int(r[1]), int(r[2]) - int(r[0])  # 计算预测框的长宽
        font_size = 0.8
        image = cv2.putText(image, text, (max(20, int(r[0])), max(20, int(r[1])-15)), cv2.FONT_HERSHEY_COMPLEX, max(font_size, 0.3), (0, 255, 255), 2)   # max()为了确保字体不过界
    return image