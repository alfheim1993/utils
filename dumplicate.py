import cv2, os
import numpy as np
from skimage.measure import compare_ssim
def remove_simillar_picture_by_perception_hash(path):
    img_list = os.listdir(path)
    hash_dic = {}
    hash_list = []
    count_num = 0
    for img_name in img_list:# break
        try:
            img = cv2.imread(os.path.join(path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            count_num+=1
            print(count_num)
        except:
            continue


        img = cv2.resize(img,(8,8))

        avg_np = np.mean(img)
        img = np.where(img>avg_np,1,0)
        hash_dic[img_name] = img
        if len(hash_list)<1:
            hash_list.append(img)
        else:
            for i in hash_list:
                flag = True
                dis = np.bitwise_xor(i,img)

                if np.sum(dis) < 3:
                    flag = False
                    os.remove(os.path.join(path, img_name))
                    if os.path.exists(os.path.join(path, img_name.replace('jpg','json'))):
                        os.remove(os.path.join(path, img_name.replace('jpg','json')))
                    break
            if flag:
                hash_list.append(img)

def remove_simillar_image_by_ssim(path):
    img_list = os.listdir(path)
    img_list.sort()
    hash_dic = {}
    save_list = []
    count_num = 0
    for i in range(len(img_list)):
        try:
            img = cv2.imread(os.path.join(path, img_list[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(256, 256))
            count_num+=1
        except:
            continue
        if count_num==1:
            save_list.append(img_list[i])
            continue
        elif len(save_list) <8:
            flag = True
            for j in range(len(save_list)):
                com_img = cv2.imread(os.path.join(path,save_list[j]))
                com_img = cv2.cvtColor(com_img,cv2.COLOR_BGR2GRAY)
                com_img = cv2.resize(com_img,(256,256))
                sim = compare_ssim(img,com_img)
                if sim > 0.4:
                    os.remove(os.path.join(path,img_list[i]))
                    flag = False
                    break
            if flag:
                save_list.append(img_list[i])
        else:
            for save_img in save_list[-5:]:
                com_img = cv2.imread(os.path.join(path,save_img))
                com_img = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)
                com_img = cv2.resize(com_img, (256, 256))
                sim = compare_ssim(img,com_img)
                if sim > 0.4:
                    os.remove(os.path.join(path,img_list[i]))
                    flag = False
                    break
            if flag:
                save_list.append(img_list[i])
if __name__=='__main__':
    path = r'E:\0-data\2-Industry\bur\knj\20200226\labeled\STAIN'
    remove_simillar_picture_by_perception_hash(path)