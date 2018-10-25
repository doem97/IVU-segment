import os
import fnmatch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import ISIC2018_dataset as ISIC
import models
from metrics import dice_loss, jacc_loss, jacc_coef, dice_jacc_mean
from isic_utils import random_crop_image, save_pred_images, save_chw_image
import pdb

# Base Environment
project_path = "/home/zichen/segment/"
K.set_image_dim_ordering('th')  # Theano dimension ordering: (channels, width, height)
np.random.seed(4)
seed = 1

# Model Settings
pre_model = None #project_path + "weights/{}.h5".format('test')
model_type = 'dilated_unet'
model_filename = project_path + "weights/temp.h5" # Suggest that everytime a new model trained as temp.h5 and be renamed later.
custom_loss = dice_loss
optimizer = Adam(lr=1e-5)
custom_metric = [jacc_coef]

# Training Control
do_stage = 1    #Stage1: only training set is provided.
                #Stage2: trainging/validation set are provided.
                #Stage3: training/validation/test set are provided.
batch_size = 32
initial_epoch = 0
n_epoch = 500
fc_size = 4000

do_resize = True
do_train = True
do_evaluate = True
do_test = True
height = 128
width = 128
channels = 3
remove_mean_imagenet = False
remove_mean_dataset = False
remove_mean_samplewise = False
rescale_mask = True # Attention: rescale_mask must be true cause in later evaluation, it will multiple 255
datasets_div = 'manu' # 'isic', 'manu' for division type
year = "2017"


# Creat image lists from training/validation/test
if datasets_div == 'isic':
    if year is "2017":
        tr_folder = project_path + "datasets/2017/ISIC-2017_Training_Data"
        tr_mask_folder = project_path + "datasets/2017/ISIC-2017_Training_Part1_GroundTruth"
        tr_csv_file = project_path + "datasets/2017/ISIC-2017_Training_Part3_GroundTruth.csv"
        val_folder = project_path + "datasets/2017/ISIC-2017_Validation_Data"
        val_mask_folder = project_path + "datasets/2017/ISIC-2017_Validation_Part1_GroundTruth"
        val_csv_file = project_path + "datasets/2017/ISIC-2017_Validation_Part3_GroundTruth.csv"
        te_folder = project_path + "datasets/2017/ISIC-2017_Test_v2_Data"
        te_mask_folder = project_path + "datasets/2017/ISIC-2017_Test_v2_Part1_GroundTruth"
        te_csv_file = project_path + "datasets/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv"
        resized_image_folder = project_path + "datasets/2017/resized"

    if year is "2018":
        tr_folder = project_path + "datasets/2018/ISIC-2018_Training_Data"
        tr_mask_folder = project_path + "datasets/2018/ISIC-2018_Training_Part1_GroundTruth"
        val_folder = project_path + "datasets/2018/ISIC-2018_Validation_Data"
        val_mask_folder = project_path + "datasets/2018/ISIC-2018_Validation_Part1_GroundTruth"
        te_folder = project_path + "datasets/2018/ISIC-2018_Test_v2_Data"
        te_mask_folder = project_path + "datasets/2018/ISIC-2018_Test_v2_Part1_GroundTruth"
        resized_image_folder = project_path + "datasets/2018/resized"

elif datasets_div == 'manu':
    tr_folder = project_path + "datasets/manual/Training"
    tr_mask_folder = project_path + "datasets/manual/Training_GroundTruth"
    val_folder = project_path + "datasets/manual/Validation"
    val_mask_folder = project_path + "datasets/manual/Validation_GroundTruth"
    te_folder = project_path + "datasets/manual/Test"
    te_mask_folder = project_path + "datasets/manual/Test_GroundTruth"
    resized_image_folder = project_path + "datasets/manual/resized"

pred_mask_folder = project_path + "pred_picture/"
pred_val_mask_folder = pred_mask_folder + "validation"
pred_te_mask_folder = pred_mask_folder + "test"

def myGenerator(image_generator, mask_generator):
    while True:
        image_gen = next(image_generator)
        mask_gen = next(mask_generator)
        image_gen = image_gen/127.5 - 1
        yield (image_gen, mask_gen)


if pre_model:
    model = load_model(pre_model, custom_objects={'dice_loss': dice_loss, 'jacc_coef': jacc_coef})
    print "Loaded previous model.\n"
    model.compile(optimizer=optimizer, loss={
                'conv7': custom_loss,
                'fc2': 'categorical_crossentropy',
              },
              loss_weights = {
                'conv7': 1.,
                'fc2': 0.,
              },
              metrics={
                'conv7': custom_metric,
                #'fc2': 'categorical_accuracy',
              })
    monitor_metric = 'val_conv7_jacc_coef' #only when multi-output it will be like ._conv7_.
elif model_type == 'unet':
    model = models.Unet(height, width, custom_loss=custom_loss, optimizer=optimizer, custom_metrics=custom_metric, fc_size=fc_size, channels=channels)
    monitor_metric = 'val_jacc_coef'
elif model_type == 'unet2':
    model = models.Unet2(height, width, custom_loss=custom_loss, optimizer=optimizer, custom_metrics=custom_metric, fc_size=fc_size, channels=channels)
    monitor_metric = 'val_jacc_coef'
elif model_type =='vgg':
    VGG16_WEIGHTS_NOTOP = project_path + 'pretrained_weights/vgg16_notop.h5'
    model = models.VGG16(height, width, pretrained=VGG16_WEIGHTS_NOTOP, freeze_pretrained = False, custom_loss = custom_loss, optimizer = optimizer, custom_metrics = custom_metric)
    monitor_metric = 'val_jacc_coef'
elif model_type == 'dilated_unet':
    model = models.dilated_unet(height, width, custom_loss = custom_loss, optimizer = optimizer, custom_metrics = custom_metric)

if do_stage == 1:
    split_ratio = [4, 1, 1]
    tr_list, val_list, te_list = ISIC.split_isic_train(tr_folder, split_ratio)
    val_folder = tr_folder
    val_mask_folder = tr_mask_folder
    te_folder = tr_folder
    te_mask_folder = tr_mask_folder
    base_tr_folder = resized_image_folder + "/train_{}_{}".format(height, width)
    base_val_folder = resized_image_folder + "/validation_{}_{}".format(height, width)
    base_te_folder = resized_image_folder + "/test_{}_{}".format(height, width)

elif do_stage == 2:
    # tr and val are same with do_split_train_data= True, test is change to val. This is for second stage.
    validation_folder = val_folder
    te_folder = val_folder
    te_mask_folder = val_mask_folder
    te_list = fnmatch.filter(os.listdir(validation_folder), '*.jpg')
    base_te_folder = resized_image_folder + "/test_{}_{}".format(height, width)

    split_ratio = [7, 1, 0]
    tr_list, val_list, false_telist = ISIC.split_isic_train(tr_folder, split_ratio)
    val_folder = tr_folder #here change 
    val_mask_folder = tr_mask_folder
    base_tr_folder = resized_image_folder + "/train_{}_{}".format(height, width)
    base_val_folder = resized_image_folder + "/validation_{}_{}".format(height, width)
    
    
elif do_stage == 3:
    tr_list = fnmatch.filter(os.listdir(tr_folder), '*.jpg') 
    val_list = fnmatch.filter(os.listdir(val_folder), '*.jpg')
    te_list = fnmatch.filter(os.listdir(te_folder), '*.jpg')
    base_tr_folder = resized_image_folder + "/train_{}_{}".format(height, width)
    base_val_folder = resized_image_folder + "/validation_{}_{}".format(height, width)
    base_te_folder = resized_image_folder + "/test_{}_{}".format(height, width)


# Check folder which stored the resized images
base_tr_image_folder = os.path.join(base_tr_folder, "image")
base_tr_mask_folder = os.path.join(base_tr_folder, "mask")
   
base_val_image_folder = os.path.join(base_val_folder, "image")
base_val_mask_folder = os.path.join(base_val_folder, "mask")
    
base_te_image_folder = os.path.join(base_te_folder, "image")
base_te_mask_folder = os.path.join(base_te_folder, "mask")
 
# Resize images and restore resized images to new folder
if do_resize:
    print("Begin resizing Images...")
    if not os.path.exists(base_tr_folder):
        ISIC.resize_image(tr_folder, tr_mask_folder, tr_list, base_tr_image_folder, 
                base_tr_mask_folder, height, width)
    if not os.path.exists(base_val_folder):
        ISIC.resize_image(val_folder, val_mask_folder, val_list, base_val_image_folder, 
                base_val_mask_folder, height, width)
    if not os.path.exists(base_te_folder):  
        ISIC.resize_image(te_folder, te_mask_folder, te_list, 
                base_te_image_folder, base_te_mask_folder, height, width) 

[val_image, val_mask] = ISIC.load_image(
        base_val_image_folder, base_val_mask_folder,
        val_list, height, width, remove_mean_imagenet, 
        remove_mean_dataset,rescale_mask)

val_image = val_image/127.5 - 1

if do_train:
    # Data augmentation and generation
    if not os.path.exists(base_tr_folder):
        print "No prepared training data"

    [tr_image, tr_mask] = ISIC.load_image(
            base_tr_image_folder, base_tr_mask_folder,
            tr_list, height, width, remove_mean_imagenet, 
            remove_mean_dataset,rescale_mask)

    data_gen_args = dict(
          featurewise_center=False,
          samplewise_center=remove_mean_samplewise,
          featurewise_std_normalization=False,
          samplewise_std_normalization=False,
          zca_whitening=False,
          rotation_range=270,
          width_shift_range=0.1,
          height_shift_range=0.1,
          zoom_range=0.2,
          channel_shift_range=0,
          shear_range=0.2,
          fill_mode='reflect', 
          horizontal_flip=False,
          vertical_flip=False,
          preprocessing_function=random_crop_image,
          dim_ordering=K.image_dim_ordering())
    data_gen_mask_args = dict(
            data_gen_args.items() + {'fill_mode':'nearest',
                'samplewise_center':False,
                'featurewise_center':False,
                'featurewise_std_normalization':False,
                'samplewise_std_normalization':False,
                }.items(),)
    image_augment = ImageDataGenerator(**data_gen_args)
    mask_augment = ImageDataGenerator(**data_gen_mask_args)
    image_augment.fit(tr_image, augment=True, seed=1)
    
    tr_image_generator = image_augment.flow(tr_image, batch_size=batch_size, seed=seed)
    tr_mask_generator = mask_augment.flow(tr_mask, batch_size=batch_size, seed=seed)


    tr_data_generator = myGenerator(tr_image_generator, tr_mask_generator)

    # Check model
    model_checkpoint = ModelCheckpoint(model_filename, monitor=monitor_metric, 
            save_best_only=True, verbose=1)

    # Training
    n_samples = len(tr_list)
    history = model.fit_generator(tr_data_generator,
            samples_per_epoch=n_samples,
            nb_epoch=n_epoch,
            validation_data=(val_image, val_mask),
            callbacks=[model_checkpoint],
            initial_epoch=initial_epoch)
    train = None; train_mask = None 
    
# Evaluate model: evaluate val 

if do_evaluate:
    model.load_weights(model_filename)
    val_pred_mask = model.predict(val_image)
    save_pred_images(val_pred_mask, val_list, pred_val_mask_folder)
    for pixel_threshold in [0.5]: # np.arange(0.3, 1, 0.05):
        val_pred_mask = np.where(val_pred_mask>=pixel_threshold, 1, 0)
        val_pred_mask = val_pred_mask * 255
        val_pred_mask = val_pred_mask.astype(np.uint8)
        print "Validation Prediction Max:{}, Min:{}".format(np.max(val_pred_mask), 
                np.min(val_pred_mask))
        print model.evaluate(val_image, val_mask, batch_size = batch_size, verbose=1)
        dice, jacc = dice_jacc_mean(val_mask, val_pred_mask, smooth = 0)
        print "Resized val dice coef: {:.4f}".format(dice)
        print "Resized val jacc coef: {:.4f}".format(jacc)

if do_test:
    [te_image, te_mask] = ISIC.load_image(
            base_te_image_folder, base_te_mask_folder,
            te_list, height, width, remove_mean_imagenet, 
            remove_mean_dataset, rescale_mask)
    te_image = te_image/127.5 - 1
    model.load_weights(model_filename)
    te_pred_mask = model.predict(te_image)
    save_pred_images(te_pred_mask, te_list, pred_te_mask_folder)
    for pixel_threshold in [0.5]: # np.arange(0.3, 1, 0.05):
        te_pred_mask = np.where(te_pred_mask>=pixel_threshold, 1, 0)
        te_pred_mask = te_pred_mask * 255
        te_pred_mask = te_pred_mask.astype(np.uint8)
        print "Test Prediction Max:{}, Min:{}".format(np.max(te_pred_mask), 
                np.min(te_pred_mask))
        print model.evaluate(te_image, te_mask, batch_size = batch_size, verbose=1)
        dice, jacc = dice_jacc_mean(te_mask, te_pred_mask, smooth = 0)
        print "Resized te dice coef: {:.4f}".format(dice)
        print "Resized te jacc coef: {:.4f}".format(jacc)