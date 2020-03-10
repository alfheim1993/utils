import json, os
import cv2
import numpy as np
import pandas as pd


class Mask2CoCo:
	def __init__(self, code_list, rep):
		self.images = []
		self.annotations = []
		self.categories = []
		self.img_id = 0
		self.ann_id = 0
		self.code_list = code_list
		self.rep = rep

	# 读取json文件，返回一个json对象
	def read_jsonfile(self, path):
		with open(path, "r", encoding='utf-8') as f:
			return json.load(f)


	def save_coco_json(self, instance, save_path):
		json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

	#  由json文件构建COCO
	def to_coco(self, im_paths_all):
		classname_to_id = dict(zip(self.code_list, range(1, len(self.code_list) + 1)))
		self._init_categories(classname_to_id)
		for i, jpg_path in enumerate(im_paths_all):
			label = os.path.basename(os.path.dirname(jpg_path))
			if label not in self.code_list:
				continue
			print(i, jpg_path, label)
			mask_path = jpg_path.replace('.jpg', self.rep)
			if os.path.exists(mask_path):
				self.images.append(self._image(jpg_path))
				mask = cv2.imread(mask_path)
				mask = cv2.bitwise_not(mask)
				mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
				ret, thresh = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
				im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			else:
				print('no mask')
				continue
			for ob in contours:
				annotation = self._annotation(ob, label, classname_to_id)
				self.annotations.append(annotation)
				self.ann_id += 1
			self.img_id += 1
		instance = dict()
		instance['info'] = 'javis created'
		instance['license'] = ['license']
		instance['images'] = self.images
		instance['annotations'] = self.annotations
		instance['categories'] = self.categories
		return instance

	# 构建类别
	def _init_categories(self, classname_to_id):
		for k, v in classname_to_id.items():
			category = dict()
			category['id'] = v
			category['name'] = k
			self.categories.append(category)

	# 构建COCO的image字段
	def _image(self, path):
		image = {}
		im = cv2.imread(path)
		image['width'] = im.shape[1]
		image['height'] = im.shape[0]
		image['id'] = self.img_id
		image['file_name'] = path
		return image

	# 构建COCO的annotation字段
	def _annotation(self, ob, label, classname_to_id):
		area_ = cv2.contourArea(ob)

		annotation = {}
		annotation['id'] = self.ann_id
		annotation['image_id'] = self.img_id
		annotation['category_id'] = classname_to_id[label]

		annotation['segmentation'] = [ob.flatten().tolist()]
		# print(annotation['segmentation'])
		annotation['bbox'] = self._get_box(ob)
		annotation['iscrowd'] = 0
		annotation['area'] = area_
		return annotation

	# COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
	def _get_box(self, ob):
		min_x = min_y = np.inf
		max_x = max_y = 0
		for point in ob:
			x, y = point[0][0], point[0][1]
			min_x = min(min_x, x)
			min_y = min(min_y, y)
			max_x = max(max_x, x)
			max_y = max(max_y, y)
		return [int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)]


def train_test_split(data, test_size=0.1):
	n_val = int(len(data) * test_size)
	np.random.shuffle(data)
	train_data = data[:-n_val]
	val_data = data[-n_val:]
	return train_data, val_data


def generate_coco(base_dir, code_dirname, code_list, rep, copy=1, test_size=0.1):
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
	# 获取全部图片路径和类别
	import glob
	jpg_list_path, answer = [], []
	for code_dir in code_list:
		code_path = os.path.join(ori_path, code_dir)
		jpg_path = glob.glob(code_path + "/*.jpg")
		answer.extend([code_dir] * len(jpg_path))
		jpg_list_path.extend(jpg_path)
	# 保存答案csv
	data = pd.DataFrame({'jpg_path': jpg_list_path, 'codes': answer})
	data.to_csv('answer.csv', index=None)

	# 拆分数据集
	np.random.seed(41)
	train_path, val_path = train_test_split(jpg_list_path, test_size=test_size)
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
	l2c_train = Mask2CoCo(code_list, rep)
	train_instance = l2c_train.to_coco(train_path)
	l2c_train.save_coco_json(train_instance, train_ann_path)

	# 把验证集转化为COCO的json格式
	l2c_val = Mask2CoCo(code_list, rep)
	val_instance = l2c_val.to_coco(val_path)
	l2c_val.save_coco_json(val_instance, val_ann_path)


if __name__ == '__main__':
	# 修改基础设置
	base_dir = r"D:\2-deep_learning\data\adc\4350\43B01"
	code_dirname = 'mask'
	copy = 0
	test_size = 0.1
	code_list = ['TEINR', 'TPDPS']
	rep = '_mask.png'
	generate_coco(base_dir, code_dirname, code_list, rep, copy, test_size)
