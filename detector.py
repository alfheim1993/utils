import mmcv
from mmdet.apis import init_detector, inference_detector
import pycocotools.mask as maskUtils
import glob, os, json
import numpy as np
import base64,cv2


class Detector:
	def __init__(self, config_file, checkpoint_file, class_names):
		self.config_file = config_file
		self.checkpoint_file = checkpoint_file
		self.model = self._load_model()
		self.class_names = class_names

	def _load_model(self):
		print('loading model...')
		model = init_detector(self.config_file, self.checkpoint_file, device='cuda:0')
		print('loading complete!')
		return model

	def save_res_img(self, img, labels, bboxes, score_thr=0, out_file=None):
		mmcv.imshow_det_bboxes(img.copy(), bboxes, labels, show=False, class_names=self.class_names, score_thr=score_thr, out_file=out_file)

	def detect_all(self, imgs, thres, save_num, save_path):
		result = []
		for i, img in enumerate(imgs):
			detect_res = inference_detector(self.model, img)
			print(i, imgs[i])
			img = mmcv.imread(imgs[i])
			img = img.copy()
			if isinstance(detect_res, tuple):
				bbox_result, segm_result = detect_res
			else:
				bbox_result, segm_result = detect_res, None
			bboxes = np.vstack(bbox_result)
			labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
			labels = np.concatenate(labels)
			if segm_result is not None:
				segms = mmcv.concat_list(segm_result)
				inds = np.where(bboxes[:, -1] > thres)[0]
				for i in inds:
					color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
					mask = maskUtils.decode(segms[i]).astype(np.bool)
					img[mask] = img[mask] * 0.5 + color_mask * 0.5
			if len(bboxes) > 0:
				if i < save_num:
					self.save_res_img(img, labels, bboxes, thres, out_file=os.path.join(save_path + os.path.basename(imgs[i])))
				for j, bbox in enumerate(bboxes):
					if float(bbox[4]) > thres:
						res_dic = {'name': os.path.basename(imgs[i]), 'category': self.class_names[labels[j]], 'bbox': [round(float(x), 2) for x in bbox[:4]], 'score': float(bbox[4])}
						print(res_dic)
						result.append(res_dic)
		return result

	def detect_maxone(self, imgs, thres, save_num, save_path):
		result = []
		for i, img in enumerate(imgs):
			detect_res = inference_detector(self.model, img)
			print(i, imgs[i])
			img = mmcv.imread(imgs[i])
			img = img.copy()
			if isinstance(detect_res, tuple):
				bbox_result, segm_result = detect_res
			else:
				bbox_result, segm_result = detect_res, None
			bboxes = np.vstack(detect_res)
			labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
			labels = np.concatenate(labels)
			if segm_result is not None:
				segms = mmcv.concat_list(segm_result)
				masks = maskUtils.decode(segms)
				inds = np.where(bboxes[:, -1] > thres)[0]
				for i in inds:
					color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
					mask = maskUtils.decode(segms[i]).astype(np.bool)
					img[mask] = img[mask] * 0.5 + color_mask * 0.5
			if len(bboxes) > 0:
				scores = bboxes[:, 4]
				idx = np.where(scores == np.max(scores))[0][0]
				if float(scores[idx]) > thres:
					if i < save_num:
						self.save_res_img(img, labels, bboxes, thres, out_file=os.path.join(save_path + os.path.basename(imgs[i])))
					if segm_result is not None:
						res_dic = {'name': os.path.basename(imgs[i]), 'category': self.class_names[labels[idx]],
						           'bbox': [round(float(x), 2) for x in bboxes[idx][:4]], 'score': float(bboxes[idx][4]),
						           'mask': masks[:, :, idx]}
					else:
						res_dic = {'name': os.path.basename(imgs[i]), 'category': self.class_names[labels[idx]],
						           'bbox': [round(float(x), 2) for x in bboxes[idx][:4]],
						           'score': float(bboxes[idx][4])}
					print(res_dic)
					result.append(res_dic)
			else:
				res_dic = {'name': os.path.basename(imgs[i]), 'category': 'None',
						           'bbox': [0,0,0,0],
						           'score': 1.0}
				print(res_dic)
				result.append(res_dic)
		return result

	def generate_label_file(self, imgs, thres):
		for i, img in enumerate(imgs):
			detect_res = inference_detector(self.model, img)
			print(i, imgs[i])
			img = mmcv.imread(imgs[i])
			img = img.copy()
			if isinstance(detect_res, tuple):
				bbox_result, segm_result = detect_res
			else:
				bbox_result, segm_result = detect_res, None
			bboxes = np.vstack(bbox_result)
			labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
			labels = np.concatenate(labels)
			if segm_result is not None:
				segms = mmcv.concat_list(segm_result)
				inds = np.where(bboxes[:, -1] > thres)[0]
				for i in inds:
					color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
					mask = maskUtils.decode(segms[i]).astype(np.bool)
					img[mask] = img[mask] * 0.5 + color_mask * 0.5
			shapes = []
			if len(bboxes) > 0:
				for j, bbox in enumerate(bboxes):
					shape = {
						"label": self.class_names[labels[j]],
						"points": [
							[
								float(bbox[0]),
								float(bbox[1])
							],
							[
								float(bbox[2]),
								float(bbox[3])
							]
						],
						"group_id": None,
						"shape_type": "rectangle",
						"flags": {}
					}
					if bbox[4] > thres:
						shapes.append(shape)
			img = cv2.imread(imgs[i])
			img_str = cv2.imencode('.jpg', img)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
			b64_code = base64.b64encode(img_str)
			label_file = {
				"version": "4.2.7",
				"flags": {},
				"shapes": shapes,
				"imagePath": os.path.basename(imgs[i]),
				"imageData": b64_code.decode(),
				"imageHeight": img.shape[0],
				"imageWidth": img.shape[1]
			}
			with open(imgs[i].replace('.jpg', '.json'), 'w') as f:
				json.dump(label_file, f)

	def save_result(self, result, out_name, file_type='json'):
		if file_type == 'json':
			with open('result/' + out_name+'.'+file_type, 'w', encoding='utf-8') as fp:
				json.dump(result, fp, indent=4, separators=(',', ': '))
		if file_type == 'csv':
			with open('result/' + out_name+'.'+file_type, 'w',encoding='utf-8-sig') as fp:
				fp.write(','.join(['name', 'category', 'score'])+'\n')
				for res_dic in result:
					fp.write(','.join([res_dic['name'], res_dic['category'], str(res_dic['score'])])+'\n')
if __name__ == '__main__':
	config_file = 'config-fasterrcnn.py'
	checkpoint_file = 'models-20200226/epoch_30.pth'
	with open('codes.json', 'r') as f:
		class_names = list(json.load(f).values())
	detctor = Detector(config_file, checkpoint_file, class_names)

	# 测试多张图片
	data_path = '/mnt/data/knj/coco/test'
	imgs = glob.glob(data_path + '/*.jpg')
	save_path = './result/img/'
	thres = 0.5
	save_num = 20
	# result = detctor.detect_maxone(imgs, thres, save_num, save_path)
	# out_name = os.path.basename(data_path)
	# detctor.save_result(result, out_name, 'csv')
	detctor.generate_label_file(imgs, 0.5)
	print('over!')
