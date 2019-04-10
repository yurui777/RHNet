import tensorflow as tf
import numpy as np
import time
import os
import RHNet as model
import cv2
import math
import utils


output_dir = './output/'
test_path = '/home/xu519/yurui/ProcessedData/QNRF/test/img'
test_label_path = '/home/xu519/yurui/ProcessedData/QNRF/test/den'
model_path='/home/xu519/yurui/tf-conut/QNRF_result/1722.ckpt'
input_img=tf.placeholder(tf.float32,shape=[1,None,None,1])
input_den=tf.placeholder(tf.float32,shape=[1,None,None,1])

pre_den=model.create(input_img)
loss=tf.losses.mean_squared_error(pre_den,input_den)

init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
summary=tf.summary.merge_all()

data_loader = utils.ImageDataLoader(test_path, test_label_path, shuffle=True, gt_downsample=True, pre_load=True)


config=tf.ConfigProto(allow_soft_placement=True)
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    saver=tf.train.Saver()
    saver.restore(sess,model_path)
    mae,mse=0,0
    num_test=0
    # Loop through all the images.
    for blob in data_loader:
        num_test+=1
        # Read the image and the ground truth
        test_image_r=blob['data']
        test_density_r=blob['gt_density']
        tru_count=np.sum(test_density_r)
        # Prepare the feed_dict
        feed_dict_data = {
            input_img: test_image_r,
            input_den: test_density_r,
        }
        test_den = sess.run([pre_den], feed_dict=feed_dict_data)
        pre_count=np.sum(test_den)
        mae+=abs(pre_count-tru_count)
        mse+=(pre_count-tru_count)*(pre_count-tru_count)


    mae=mae/num_test
    mse=np.sqrt(mse/num_test)

    print('mae:%.4f mse:%.4f '%(mae,mse))




