import json
import cv2
import numpy as np
import os
import pandas as pd


class Labelme2CoCo:

    def __init__(self, classname_to_id, id_to_classname, label_from_dir, type='jpg'):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.classname_to_id = classname_to_id
        self.id_to_classname = id_to_classname
        self.label_from_dir = label_from_dir
        self.type = type

    # 读取json文件，返回一个json对象
    @staticmethod
    def read_jsonfile(path):
        with open(path, "r", encoding='utf-8') as f:
            print(path)
            data = json.load(f)
            return data

    @staticmethod
    def save_coco_json(instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, jpg_list_path):
        # 修改categories属性
        self._init_categories()
        for i, jpg_path in enumerate(jpg_list_path):
            # 看是否存在标注文件
            json_path = jpg_path.replace('.' + self.type, '.json')
            if not os.path.exists(json_path):
                continue
            # 获取json内容
            obj = self.read_jsonfile(json_path)
            # 添加图片路径到images属性
            self.images.append(self._image(jpg_path))
            # 获取标注信息，修改annotations，ann_id，img_id属性
            shapes = obj['shapes']
            for shape in shapes:
                if self.label_from_dir:
                    label = os.path.basename(os.path.dirname(json_path))
                else:
                    label = shape['label']
                print(i, json_path, label)
                annotation = self._annotation(shape, label)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        # 更新instance全部属性
        instance = dict()
        instance['info'] = 'zwp created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for _id, name in self.id_to_classname.items():
            category = dict()
            category['id'] = _id
            category['name'] = name
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, jpg_path):
        image = {}
        im = cv2.imread(jpg_path)
        image['width'] = im.shape[1]
        image['height'] = im.shape[0]
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(jpg_path)
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, label):
        points = [[int(x), int(y)] for x, y in shape['points']]
        if shape['shape_type'] == 'rectangle':
            points = [[points[0][0], points[0][1]],
                      [points[1][0], points[0][1]],
                      [points[1][1], points[1][1]],
                      [points[0][0], points[1][1]]]
        area_ = cv2.contourArea(np.expand_dims(points, 1))
        segmentation = [np.concatenate(points).tolist()]

        annotation = dict()
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        if self.classname_to_id[label]<=0 or self.classname_to_id[label]>=8:
            print(1)
        annotation['category_id'] = self.classname_to_id[label]
        annotation['segmentation'] = segmentation
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = area_
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    @staticmethod
    def _get_box(points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


def train_test_split(data, size=0.1):
    n_val = int(len(data) * size)
    np.random.shuffle(data)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    return train_data, val_data


def generate_coco(base_dir, code_dirname, copy=1, test_size=0.1, label_from_dir=1, type='jpg', classname_to_id=None, id_to_classname=None):
    os.chdir(base_dir)
    # 创建文件夹
    if not os.path.exists('coco'):
        os.makedirs(r'coco\annotations')
        os.makedirs(r'coco\images')
        os.makedirs(r'coco\test')
    # 定义路径
    ori_path = code_dirname
    train_ann_path = r'coco\annotations\instances_train.json'
    val_ann_path = r'coco\annotations\instances_val.json'
    images_path = r'coco\images'
    test_images_path = r'coco\test'
    # 得到类别以及生成id
    code_dirs = [code for code in os.listdir(ori_path)]
    if classname_to_id is None:
        classname_to_id = dict(zip(code_dirs, range(1, len(code_dirs) + 1)))
        id_to_classname =dict(zip(classname_to_id.values(), classname_to_id.keys()))
    with open('codes.json', 'w') as f:
        json.dump(id_to_classname, f)

    # 获取全部图片路径和类别
    import glob
    jpg_list_path, answer = [], []
    for code_dir in code_dirs:
        code_path = os.path.join(ori_path, code_dir)
        jpg_path = glob.glob(code_path + "/*." + type)
        answer.extend([code_dir] * len(jpg_path))
        jpg_list_path.extend(jpg_path)
    # 保存答案csv
    data = pd.DataFrame({'name': [os.path.basename(x) for x in jpg_list_path], 'codes': answer})
    data.to_csv('answer.csv', index=None)

    # 拆分数据集
    np.random.seed(41)
    train_path, val_path = train_test_split(jpg_list_path, size=test_size)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # 复制图片
    import shutil
    if copy:
        print('copy start!')
        for p in train_path + val_path:
            print(p)
            shutil.copy(p, os.path.join(images_path, os.path.basename(p)))
        for p in val_path:
            print(p)
            shutil.copy(p, os.path.join(test_images_path, os.path.basename(p)))

    # 把训练集转化为COCO的json格式
    l2c_train = Labelme2CoCo(classname_to_id, id_to_classname, label_from_dir, type)
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, train_ann_path)

    # 把验证集转化为COCO的json格式
    l2c_val = Labelme2CoCo(classname_to_id, id_to_classname, label_from_dir, type)
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, val_ann_path)


if __name__ == '__main__':
    # 修改基础设置
    base_dir = r"E:\0-data\2-Industry\bur\knj\mix-0206-0226"
    code_dirname = 'labeled'
    copy = True
    test_size = 0.01
    label_from_dir = False
    if not label_from_dir:
        classname_to_id = {'Dirty': 1, 'STAIN': 1, 'BROKEN': 2, 'BURR': 3, 'TFT BALI': 3, 'CF BALI': 3,
                           'BLACK_BURR': 3, 'CHIP': 4, 'TFT Chipping': 4, 'CF Chipping': 4, 'CRACK': 5}
        id_to_classname = {1: 'STAIN', 2: 'BROKEN', 3: 'BURR', 4: 'CHIP', 5: 'CRACK'}
    else:
        classname_to_id = None
        id_to_classname = None
    type = 'jpg'

    generate_coco(base_dir, code_dirname, copy, test_size, label_from_dir, type, classname_to_id, id_to_classname)





