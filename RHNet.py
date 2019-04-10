import tensorflow as tf
import numpy as np

def create(input,bn=False):
    tf.summary.image('input',input,1)
    if bn:
        input=tf.cast(input,tf.float32)*(1./255)-0.5
    net_1=block1(input)
    net_2=block2(net_1)

    return net_2


def block1(input):
    with tf.variable_scope('block1'):
        conv_1=conv(input,name='conv_1',kernel_size=5,filters=24)
        pool_1=pool(conv_1,name='pool_1',kernel_size=2,strides=2)
        conv_2=conv(pool_1,name='conv_2',kernel_size=3,filters=48)
        pool_2=pool(conv_2,name='pool_2',kernel_size=2,strides=2)
        conv_3=conv(pool_2,name='conv_3',kernel_size=3,filters=24)
        conv_4=conv(conv_3,name='conv_4',kernel_size=3,filters=12)
        return conv_4

def block2(input):
    # this block contains 4 dilated convs
    with tf.variable_scope('block2'):
        conv_dia_1=conv_dila(input,'cov_dia_1',3,12,rate=2)
        conv_dia_2 = conv_dila(input, 'cov_dia_2', 3, 10, rate=4)
        conv_dia_3 = conv_dila(input, 'cov_dia_3', 3, 8, rate=6)
        conv_dia_4 = conv_dila(input, 'cov_dia_4', 3, 6, rate=8)
        res=tf.concat([conv_dia_1,conv_dia_2,conv_dia_3,conv_dia_4],axis=3)
        output=conv(res,'con_fusd',1,1)
        return output


def conv(input,name,kernel_size,filters,strides=1,activation=tf.nn.relu):
    input_shape=input.get_shape()[-1].value

    with tf.variable_scope(name):
        weight=tf.Variable(tf.truncated_normal([kernel_size,kernel_size,input_shape,filters],
                                               stddev=0.01),dtype=tf.float32,name='w')
        bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[filters]), dtype=tf.float32, name='b')
        conv=tf.nn.conv2d(input,weight,strides=[1,strides,strides,1],padding='SAME')
        output=activation(tf.nn.bias_add(conv,bias))
        return output



def conv_dila(input,name,kernel_size,filters,rate,strides=1,activation=tf.nn.relu):
    input_shape=input.get_shape()[-1].value
    with tf.variable_scope(name):
        weight = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, input_shape, filters],
                                                 stddev=0.01), dtype=tf.float32, name='w')
        bias=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[filters]),dtype=tf.float32,name='b')
        conv=tf.nn.atrous_conv2d(input,weight,rate=rate,padding='SAME')
        output=activation(tf.nn.bias_add(conv,bias))

        return output


def pool(input,name,kernel_size,strides):
    with tf.variable_scope(name):
        output=tf.nn.max_pool(input,ksize=[1,kernel_size,kernel_size,1],
                              strides=[1,strides,strides,1],
                              padding='SAME')
        return output