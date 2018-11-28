# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import os
import i_s

data_path = 'mnist/wgan-gp'
img_list = []
for _,_,files in os.walk(data_path):
    for file_name in files:
        if not (file_name.endswith('.jpg') or file_name.endswith('.png')):
            continue
        img_dir = os.path.join(data_path,file_name)
        img_arr = cv2.imread(img_dir)
#        img_arr = img_arr.transpose(2,0,1)
#        img_arr = np.array(img_arr.reshape((1,)+img_arr.shape))
        img_list.append(img_arr)
print('original data')
i_s.get_inception_score(img_list)


#
#data_path = 'face/face_test1'
#img_list = []
#for _,_,files in os.walk(data_path):
#    for file_name in files:
#        if not (file_name.endswith('.jpg') or file_name.endswith('.png')):
#            continue
#        img_dir = os.path.join(data_path,file_name)
#        img_arr = cv2.imread(img_dir)
#        img_arr = img_arr.transpose(2,0,1)
#        img_arr = np.array(img_arr.reshape((1,)+img_arr.shape))
#        img_list.append(img_arr)
#mat1 = np.concatenate(img_list)