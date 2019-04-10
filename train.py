import tensorflow as tf
import numpy as np
import time
import os
import RHNet as model
import cv2
import math

import utils
#Define some needed parameters

momentum=0.9
EPOCH=2000
lr=0.00001


method = 'mcnn'
dataset_name = 'WE'
output_dir = './saved_models/'
train_path = '/home/xu519/yurui/ProcessedData/QNRF/train/img'
train_label_path = '/home/xu519/yurui/ProcessedData/QNRF/train/den'
val_path = '/home/xu519/yurui/ProcessedData/QNRF/test/img'
val_label_path = '/home/xu519/yurui/ProcessedData/QNRF/test/den'

input_img=tf.placeholder(tf.float32,shape=[1,None,None,1])
input_den=tf.placeholder(tf.float32,shape=[1,None,None,1])

pre_den=model.create(input_img)
loss=tf.losses.mean_squared_error(pre_den,input_den)
optimizer=tf.train.AdamOptimizer(lr)
train_op=optimizer.minimize(loss)
default_graph=tf.get_default_graph()

init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
summary=tf.summary.merge_all()

data_loader = utils.ImageDataLoader(train_path, train_label_path, shuffle=True, gt_downsample=True, pre_load=True)
data_loader_val = utils.ImageDataLoader(val_path, val_label_path, shuffle=False, gt_downsample=True, pre_load=True)

config=tf.ConfigProto(allow_soft_placement=True)
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True

with tf.Session(graph=default_graph,config=config) as sess:

    sess.run(init)
    saver=tf.train.Saver(max_to_keep=0)
    best_mae=100000
    bast_mse=100000
    writer=tf.summary.FileWriter('./exp')
    writer.add_graph(sess.graph)
    for epo in range(1,EPOCH+1):
        start=time.time()
        num=0
        total_loss=0
        for blob in data_loader:
            num+=1
            train_image_1=blob['data']

            train_density_1=blob['gt_density']

            _,pre_density,train_loss=sess.run([train_op,pre_den,loss],feed_dict={
                input_img:train_image_1,
                input_den:train_density_1,
            })

            total_loss+=train_loss
            if num%500==0:
                count_tru=np.sum(train_density_1)
                count_pre=np.sum(pre_density)
                print('truth:'+str(count_tru)+'  pre:'+str(count_pre))

        saver.save(sess, './result/'+str(epo)+'.ckpt')

        time_one_epoch=time.time()-start
        average_loss=total_loss/num

        print(str(epo)+'epoch Training loss is: %.4f ; time: %.4f' %(average_loss,time_one_epoch))



        valid_start_time = time.time()
        mae,mse=0,0
        total_val_loss=0
        num_val=0
        # Loop through all the images.
        for blob in data_loader_val:
            num_val+=1
            # Read the image and the ground truth
            val_image_r=blob['data']
            val_density_r=blob['gt_density']
            tru_count=np.sum(val_density_r)
            # Prepare the feed_dict
            feed_dict_data = {
                input_img: val_image_r,
                input_den: val_density_r,
            }

            # Compute the loss per image
            val_loss,val_den = sess.run([loss,pre_den], feed_dict=feed_dict_data)
            pre_count=np.sum(val_den)
            mae+=abs(pre_count-tru_count)
            mse+=(pre_count-tru_count)*(pre_count-tru_count)
            # Accumalate the validation loss across all the images.
            total_val_loss = total_val_loss +val_loss
        mae=mae/num_val
        mse=np.sqrt(mse)/num_val
        if mae<best_mae:
            best_mae=mae
            best_mse=mse
            best_epoch=epo
        val_loss_1=total_val_loss/num_val
        print(str(epo)+' epoch: mae:%.4f mse:%.4f  val_loss: %.4f'%(mae,mse,val_loss_1))

        print('best epoch:'+str(best_epoch)+'best_mae:%.4f  best_mse:%.4f'%(best_mae,best_mse))



