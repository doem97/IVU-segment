import numpy as np
import cv2

def random_crop_image(image, crop_size=()):
    '''Read image of size(color_channel, width, height) and crop it with.'''
    image_temp = np.zeros(image.shape)
    height, width = image.shape[1:3]
    #random_array = np.random.random(size=4)
    random_array = [0.75,0.75,0.75,0.75]
    w = int((width*0.5)*(1+random_array[0]*0.5))
    h = int((height*0.5)*(1+random_array[1]*0.5))
    x = int(random_array[2]*(width-w))
    y = int(random_array[3]*(height-h))

    image_crop = image[0:3,y:h+y,x:w+x]
    for index, i in enumerate(image_crop):
        image_temp[index] = cv2.resize(i, (width, height))
    return image_temp

def fixed_crop_image(image, crop_size=()):
    '''Read image of size(color_channel, width, height) and crop it with.'''
    image_temp = np.zeros(image.shape)
    height, width = image.shape[1:3]
    #random_array = np.random.random(size=4)
    fixed_array = [0.75,0.75,0.75,0.75]
    w = int((width*0.5)*(1+fixed_array[0]*0.5))
    h = int((height*0.5)*(1+fixed_array[1]*0.5))
    x = int(fixed_array[2]*(width-w))
    y = int(fixed_array[3]*(height-h))

    image_crop = image[0:3,y:h+y,x:w+x]
    for index, i in enumerate(image_crop):
        image_temp[index] = cv2.resize(i, (width, height))
    return image_temp