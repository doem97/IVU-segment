import os
import fnmatch 
import numpy as np
import random
import cv2 as cv
from keras.utils.np_utils import to_categorical
import pandas as pd
np.random.seed(4)
import pdb
mean_imagenet = [123.68, 103.939, 116.779] # rgb
mean_train_isic2017 = np.array([[[ 180.71656799]],[[ 151.13494873]],[[ 139.89967346]]]);
std_train_isic2017 = train_std = np.array([[[1]],[[1]],[[ 1]]]); 

def get_labels(image_list, csv_file):
    image_list = [filename.split('.')[0] for filename in image_list]
    return to_categorical(pd.read_csv(csv_file,index_col=0).loc[image_list]['label'].values.flatten())

def split_isic_train(train_folder, split_ratio):
    imglist = fnmatch.filter(os.listdir(train_folder), '*.jpg')
   
    len_imglist = len(imglist)
    num_each = len_imglist/sum(split_ratio)

    index = list(range(len_imglist))
    #random.shuffle(index)
    
    trainlist = index[0: split_ratio[0] * num_each] 
    validationlist = index[split_ratio[0] * num_each: (split_ratio[0] +  split_ratio[1])*num_each]
    testlist = index[(split_ratio[0] + split_ratio[1])  * num_each :]
    train_list = [imglist[i] for i in trainlist]
    validation_list = [imglist[i] for i in validationlist]
    test_list = [imglist[i] for i in testlist]
    return train_list, validation_list, test_list



def resize_image(ori_folder, ori_mask_folder, ori_list, resize_image_folder, resize_mask_folder, height, width):
    #pdb.set_trace()
    os.makedirs(resize_image_folder)
    os.makedirs(resize_mask_folder)
    for imgname in ori_list:
        ori_image = cv.imread(os.path.join(ori_folder, imgname))
        ori_mask  = cv.imread(os.path.join(ori_mask_folder, imgname.replace(".jpg", "_segmentation.png")), cv.IMREAD_GRAYSCALE) 
        _, ori_mask = cv.threshold(ori_mask,127,255,cv.THRESH_BINARY)
        resize_image = cv.resize(ori_image, (height, width), interpolation=cv.INTER_CUBIC)      
          
        resize_mask  = cv.resize(ori_mask, (height, width), interpolation=cv.INTER_CUBIC)
        _, resize_mask = cv.threshold(resize_mask,127,255,cv.THRESH_BINARY)

        cv.imwrite(os.path.join(resize_image_folder, imgname.replace(".jpg", ".png")), resize_image)
        cv.imwrite(os.path.join(resize_mask_folder, imgname.replace(".jpg", ".png")), resize_mask)

def resize_only_image(ori_folder, ori_list, resize_image_folder, height, width):
    #pdb.set_trace()
    os.makedirs(resize_image_folder)
    shape_dict = {}
    for imgname in ori_list:
        ori_image = cv.imread(os.path.join(ori_folder, imgname))
        shape_dict[imgname] = (ori_image.shape[1], ori_image.shape[0])
        resize_image = cv.resize(ori_image, (height, width), interpolation=cv.INTER_CUBIC)

        cv.imwrite(os.path.join(resize_image_folder, imgname.replace(".jpg", ".png")), resize_image)
    return shape_dict

def load_image(image_folder, mask_folder, image_list, height, width, remove_mean_imagenet, remove_mean_dataset, rescale_mask):
    n_channel = 3
    img_array = np.zeros((len(image_list), n_channel, height, width), dtype=np.float32)
    img_mask_array = np.zeros((len(image_list), height, width), dtype=np.float32)
    for i in range(len(image_list)):
        image_path = os.path.join(image_folder, image_list[i].replace(".jpg",".png"))
        mask_path = os.path.join(mask_folder, image_list[i].replace(".jpg",".png"))
        img = cv.imread(image_path)
        if not os.path.exists(mask_path):
            print(image_path)
            continue
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) 
        if remove_mean_imagenet:
            for channel in [0, 1, 2]:
                img[:,:,channel] -= mean_imagenet[channel]
        img = img.transpose((2,0,1)).astype(np.float32)
        if remove_mean_dataset:
            img = (img - mean_train_isic2017)/std_train_isic2017
        img_array[i] = img

        img_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        _, img_mask_array[i] = cv.threshold(img_mask,127,255,cv.THRESH_BINARY)
        if rescale_mask:
            img_mask_array[i] = img_mask_array[i]/255.

    return (img_array, img_mask_array.astype(np.uint8).reshape((img_mask_array.shape[0],1,img_mask_array.shape[1],img_mask_array.shape[2])))

def load_only_image(image_folder, image_list, height, width, remove_mean_imagenet, remove_mean_dataset):
    n_channel = 3
    img_array = np.zeros((len(image_list), n_channel, height, width), dtype=np.float32)
    for i in range(len(image_list)):
        image_path = os.path.join(image_folder, image_list[i].replace(".jpg",".png"))
        img = cv.imread(image_path)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) 
        if remove_mean_imagenet:
            for channel in [0, 1, 2]:
                img[:,:,channel] -= mean_imagenet[channel]
        img = img.transpose((2,0,1)).astype(np.float32)
        if remove_mean_dataset:
            img = (img - mean_train_isic2017)/std_train_isic2017
        img_array[i] = img

    return img_array
