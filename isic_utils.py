import numpy as np
import cv2
import os
import sys
import random
import shutil

def random_crop_image(image):
    img_name = str(random.randint(0,65530))
    image_temp = np.zeros(image.shape)
    height, width = image.shape[1:]
    w = int((width*0.5)*(1+0.5*0.5))
    h = int((height*0.5)*(1+0.5*0.5))
    x = int(0.5*(width-w))
    y = int(0.5*(height-h))

    image_crop = image[:,y:h+y,x:w+x]
    for index, i in enumerate(image_crop):
        image_temp[index] = cv2.resize(i, (width, height))
    return image_temp #(channel, height, width)

def save_pred_images(image_package, img_list, save_path):
    if os.path.exists(save_path):
        print("the path %s already exists! It will be cleared." % save_path)
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)
    image_package = image_package.transpose((0,2,3,1))
    for index, i in enumerate(image_package):
        temp_path = save_path + "/" + img_list[index]
        i = i*255
        cv2.imwrite(temp_path, i)

def save_chw_image(image, save_path):
    image = image.transpose((1,2,0))
    image = image*255
    cv2.imwrite(save_path, image)