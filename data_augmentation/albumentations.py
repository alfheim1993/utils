import cv2
from matplotlib import pyplot as plt
import glob, os
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomBrightnessContrast, RandomGamma
)  # 图像变换函数


def augment_train(p=.5):
    return Compose([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),  # 亮度对比度
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=0, p=.75), # 平移缩放旋转
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.7),  # 颜色变换
        CLAHE(p=1, clip_limit=8),  # 对比度受限直方图均衡化
        GaussNoise(var_limit=(10, 50), always_apply=False, p=0.5),  # 高斯噪声
        Blur(blur_limit=15, p=0.1)  # 模糊
    ], p=p)

image_paths = glob.glob(r'D:\2-deep_learning\data\ironbar\train_dataset' + r'\*.jpg')
for step in range(1):
    for image_path in image_paths:
        print(step, image_path)
        image = cv2.imread(image_path, 1)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        aug = augment_train(p=1)
        img_train = aug(image=image)['image']
        cv2.imwrite(r'D:\2-deep_learning\data\ironbar\train_dataset2\\' + str(step) + '_' + os.path.basename(image_path), img_train)

def augment_test(p=.5):
    return Compose([
        CLAHE(p=1, clip_limit=8),  # 对比度受限直方图均衡化
    ], p=p)

image_paths = glob.glob(r'D:\2-deep_learning\data\ironbar\test_dataset2' + r'\*.jpg')
for image_path in image_paths:
    print(image_path)
    image = cv2.imread(image_path, 1)  # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aug = augment_test(p=1)
    img_test = aug(image=image)['image']
    cv2.imwrite(r'D:\2-deep_learning\data\ironbar\test_dataset3\\' + os.path.basename(image_path),
                img_test)

# show
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(img_test)
plt.show()