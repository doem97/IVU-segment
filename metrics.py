import numpy as np
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
from keras.objectives import categorical_crossentropy
import tensorflow as tf

smooth_default = 1.
# lambda_a = 1.
# lambda_b = 1.

# def custom_loss(y_true, y_pred):
#     all_len = len(K.eval(y_true))
#     pic_len = all_len - 3
#     return -lambda_a*dice_coef(y_true[:pic_len], y_pred[:pic_len])+lambda_b*sf_coef(y_true[pic_len:all_len], y_pred[pic_len:all_len])

# def sf_coef(y_true, y_pred):
#     y_pred = K.eval(y_pred)
    
#     temp_y_pred = np.zeros(3, dtype=np.float32)
#     true_location = np.where(y_pred == np.max(y_pred))[0][0]
#     temp_y_pred[true_location] = 1
    
#     temp_y_pred = tf.convert_to_tensor(temp_y_pred)
#     y_pred = tf.convert_to_tensor(y_pred)
#     return categorical_crossentropy(y_true, temp_y_pred)


def dice_coef(y_true, y_pred, smooth = smooth_default, per_batch = True):
    if not per_batch:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    else: 
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersec = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
        union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
        return K.mean(intersec / union)

def jacc_coef(y_true, y_pred, smooth = smooth_default):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
    
def jacc_coef_v2(y_true, y_pred, smooth = smooth_default): # Jacc here uses "under 0.65 was set to 0" strategy (avoid all-zeros)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    temp_jacc_score = (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
    temp_jacc_score = tf.cond(tf.less(temp_jacc_score, 0.65), lambda :0., lambda :temp_jacc_score)
    return temp_jacc_score
    
def jacc_loss(y_true, y_pred):
    return -jacc_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
    
def dice_jacc_single(mask_true, mask_pred, smooth = smooth_default):
    bool_true = mask_true.reshape(-1).astype(np.bool)
    bool_pred = mask_pred.reshape(-1).astype(np.bool)
    if bool_true.shape != bool_pred.shape:
        raise ValueError("Masks of different sizes.")

    bool_sum = bool_true.sum() + bool_pred.sum()
    if bool_sum == 0:
        print "Empty mask"
        return 0,0
    intersec = np.logical_and(bool_true, bool_pred).sum()
    dice = 2. * intersec / bool_sum
    jacc = jaccard_similarity_score(bool_true.reshape((1, -1)), bool_pred.reshape((1, -1)), normalize=True, sample_weight=None)
    return dice, jacc

def dice_jacc_mean(mask_true, mask_pred, smooth = smooth_default):
    dice = 0
    jacc = 0
    for i in range(mask_true.shape[0]):
        current_dice, current_jacc = dice_jacc_single(mask_true=mask_true[i],mask_pred=mask_pred[i], smooth= smooth)
        dice = dice + current_dice
        jacc = jacc + current_jacc
    dice = dice/mask_true.shape[0]
    jacc = jacc/mask_true.shape[0]
    if jacc < 0.65:
        jacc = 0
    return dice, jacc
