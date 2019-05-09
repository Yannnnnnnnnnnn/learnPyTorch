# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:55:26 2019

@author: Yan
"""

# -*- coding:utf-8 -*- 
# https://blog.csdn.net/mdjxy63/article/details/78946455
__author__ = 'xuy'

 
import os
import shutil
import random
 
root_dir = r'/home/yqs/Desktop/dogs-vs-cats-redux-kernels-edition/train/0'
output_dir = r'/home/yqs/Desktop/dogs-vs-cats-redux-kernels-edition/validation/0'

percentage = 0.2 

for root, dirs, files in os.walk(root_dir):
    number_of_files = len(os.listdir(root))
    ref_copy = int(round(percentage * number_of_files))#随机筛选20%的图片到新建的文件夹当中
    for i in range(ref_copy):
        chosen_one = random.choice(os.listdir(root))
        file_in_track = root
        file_to_copy = file_in_track + '/' + chosen_one
        if os.path.isfile(file_to_copy) == True:
            shutil.move(file_to_copy,output_dir)
