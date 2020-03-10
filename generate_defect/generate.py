import json,os,glob
import cv2
import os.path as osp
import numpy as np
from albumentations import (
    HorizontalFlip, IAAPerspective, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomBrightnessContrast, RandomGamma
)  # 图像变换函数

# 缺陷随机变化
def augment(p=.5):
    return Compose([
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),  # 亮度对比度
        ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=0,  border_mode=cv2.BORDER_CONSTANT, p=1) # 平移缩放旋转
        # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.7),  # 颜色变换
        # CLAHE(p=1, clip_limit=8),  # 对比度受限直方图均衡化
        # GaussNoise(var_limit=(10, 50), always_apply=False, p=0.5),  # 高斯噪声
        # Blur(blur_limit=15, p=0.1)  # 模糊
    ], p=p)

# 处理路径
path = 'generate_defect'
# 缺陷图片
defect_im = cv2.imread(osp.join(path, 'defect', 'ori_chip.jpg'))
# 缺陷mask
with open(osp.join(path, 'defect', 'ori_chip.json')) as f:
    mask_point = np.asarray(json.load(f)['shapes'][0]['points'])
h, w, _ = defect_im.shape
im = np.zeros((h, w), dtype="uint8")
cv2.polylines(im, [mask_point], 1, 1)
cv2.fillPoly(im, [mask_point], 255)
mask = im
# 缺陷增强
aug = augment(p=1.0)
after_aug = aug(image=defect_im, mask=mask)
defect_im_aug, mask_aug = after_aug['image'], after_aug['mask']
cv2.imwrite('defect_aug.jpg', defect_im_aug)
defect = cv2.bitwise_or(defect_im_aug,defect_im_aug,mask=mask_aug)
cv2.imwrite('defect.jpg', defect)

# 正常图片添加缺陷
normal_im = cv2.imread(osp.join(path, 'normal', 'TB951932AZ06_004809_L2_CF_CHIP_001.jpg'))
mask_inv = cv2.bitwise_not(mask_aug)
bg = cv2.bitwise_or(normal_im, normal_im, mask=mask_inv)
res = cv2.add(bg, defect)
cv2.imwrite('res.jpg', res)


