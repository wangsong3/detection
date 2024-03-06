import cv2
import numpy as np

def resize_image(image, size, letterbox_image):
    """
        对输入图像进行resize
    Args:
        size:目标尺寸
        letterbox_image: bool 是否进行letterbox变换
    Returns:指定尺寸的图像
    """
    ih, iw, _ = image.shape
    h, w = size
    if letterbox_image:
        scale = min(w/iw, h/ih)       # 缩放比例
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # 生成画布
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        # 将image放在画布中心区域-letterbox
        image_back[(h-nh)//2: (h-nh)//2 + nh, (w-nw)//2:(w-nw)//2+nw, :] = image
    else:
        image_back = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return image_back  

def img2input(img):
    img = np.transpose(img, (2, 0, 1))
    img = img/255
    return np.expand_dims(img, axis=0).astype(np.float32)  # (1,3,640,640)