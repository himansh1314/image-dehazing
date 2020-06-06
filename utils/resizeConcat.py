# -*- coding: utf-8 -*-
import os
import cv2

list_dir_HAZY = os.listdir('hazy')
list_dir_GT = os.listdir('GT')
for i in range(0,len(list_dir_GT)):
    img1 = cv2.imread('GT/{}'.format(list_dir_GT[i]))
    img2 = cv2.imread('hazy/{}'.format(list_dir_HAZY[i]))
    img1 = cv2.resize(img1, (256,256))
    img2 = cv2.resize(img2, (256,256))
    img3 = cv2.hconcat((img1, img2))
    cv2.imwrite('dataset/{}.png'.format(i), img3)
    
