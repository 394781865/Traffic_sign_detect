#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from config import config
slim = tf.contrib.slim
def prelu(inputs):
    alphas = tf.get_variable('alpha', inputs.get_shape()[-1],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg

def cls_ohem(cls_prob,label):
    loss=-(label*tf.log(cls_prob+1e-10)+(1-label)*tf.log(1-cls_prob+1e-10))
    zeros=tf.zeros_like(cls_prob,dtype=tf.float32)
    valid_inds=tf.where(label>=zeros,tf.ones_like(cls_prob,dtype=tf.float32),zeros)
    keep_num=tf.cast(tf.reduce_sum(valid_inds)*config.CLS_OHEM_RATIO,tf.int32)
    loss=loss*valid_inds
    #loss=tf.Print(loss,[loss,label,cls_prob,valid_inds,tf.shape(loss),tf.reduce_sum(valid_inds)])
    _,k_index=tf.nn.top_k(loss,keep_num)
    loss=tf.gather(loss,k_index)
    #loss=tf.Print(loss,[tf.shape(loss)])
    return tf.reduce_mean(loss)

def bbox_ohem(bbox_pred,bbox_target,label):
    zeros=tf.zeros_like(label,dtype=tf.float32)
    valid_inds=tf.where(label!=zeros,tf.ones_like(label,dtype=tf.float32),zeros)
    square_error=tf.reduce_sum(tf.square(bbox_pred-bbox_target),axis=1)
    keep_num=tf.cast(tf.reduce_sum(valid_inds)*config.CLS_OHEM_RATIO,tf.int32)
    square_error=square_error*valid_inds
    _,k_index=tf.nn.top_k(square_error,keep_num)
    square_error=tf.gather(square_error,k_index)
    return tf.reduce_mean(square_error)

def P_Net(inputs,label=None,bbox_target=None,training=True):
    with slim.arg_scope([slim.conv2d],normalizer_fn=None,weights_initializer=slim.xavier_initializer(),activation_fn=prelu,biases_initializer=tf.zeros_initializer(),padding='valid'):
        net=slim.conv2d(inputs,10,kernel_size=3,scope='conv1')
        net=slim.max_pool2d(net,[2,2],stride=2,scope='pool1')
        #print net.get_shape()

        net=slim.conv2d(net,16,kernel_size=3,scope='conv2')
        #print net.get_shape()

        net=slim.conv2d(net,32,kernel_size=3,scope='conv3')
        #print net.get_shape()

        conv4_1=slim.conv2d(net,1,kernel_size=1,scope='conv4_1',activation_fn=tf.sigmoid)
        bbox_pred=slim.conv2d(net,4,kernel_size=1,scope='conv4_2',activation_fn=None)
        if training:
            cls_prob=tf.squeeze(conv4_1, [1, 2, 3], name='cls_prob')
            bbox_pred=tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
            cls_loss=cls_ohem(cls_prob,label)
            bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
            return cls_prob,bbox_pred,cls_loss,bbox_loss
        else:
            cls_prob=tf.squeeze(conv4_1, [3], name='cls_prob')
            return cls_prob,bbox_pred


def R_Net(inputs,label=None,bbox_target=None,training=True):
    with slim.arg_scope([slim.conv2d],normalizer_fn=None,weights_initializer=slim.xavier_initializer(),activation_fn=prelu,biases_initializer=tf.zeros_initializer(),padding='valid'):
        net=slim.conv2d(inputs,28,kernel_size=3,scope='conv1')
        net=slim.max_pool2d(net,[3,3],stride=2,scope='pool1',padding='same')
        #print net.get_shape()

        net=slim.conv2d(net,48,kernel_size=3,scope='conv2')
        net=slim.max_pool2d(net,[3,3],stride=2,scope='pool2')
        #print net.get_shape()

        net=slim.conv2d(net,64,kernel_size=2,scope='conv3')
        #print net.get_shape()

        net=slim.fully_connected(net,128,normalizer_fn=None,biases_initializer=tf.zeros_initializer(),activation_fn=prelu,scope='fc1')

        net=slim.flatten(net)
        cls_prob=slim.fully_connected(net,1,normalizer_fn=None,biases_initializer=tf.zeros_initializer(),activation_fn=tf.sigmoid,scope='fc2')
        bbox_pred=slim.fully_connected(net,4,normalizer_fn=None,biases_initializer=tf.zeros_initializer(),activation_fn=None,scope='fc3')

        cls_prob=tf.squeeze(cls_prob, [1], name='cls_prob')
        if training:
            cls_loss=cls_ohem(cls_prob,label)
            bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
            return cls_prob,bbox_pred,cls_loss,bbox_loss
        else:
            return cls_prob,bbox_pred

def O_Net(inputs,label=None,bbox_target=None,training=True):
    with slim.arg_scope([slim.conv2d],normalizer_fn=None,weights_initializer=slim.xavier_initializer(),activation_fn=prelu,biases_initializer=tf.zeros_initializer(),padding='valid'):
        net=slim.conv2d(inputs,32,kernel_size=3,scope='conv1')
        net=slim.max_pool2d(net,[3,3],stride=2,scope='pool1',padding='same')
        #print net.get_shape()

        net=slim.conv2d(net,64,kernel_size=3,scope='conv2')
        net=slim.max_pool2d(net,[3,3],stride=2,scope='pool2')
        #print net.get_shape()

        net=slim.conv2d(net,64,kernel_size=3,scope='conv3')
        net=slim.max_pool2d(net,[2,2],stride=2,scope='pool3')
        #print net.get_shape()

        net=slim.conv2d(net,128,kernel_size=2,scope='conv4')
        #print net.get_shape()

        net=slim.fully_connected(net,256,normalizer_fn=None,biases_initializer=tf.zeros_initializer(),activation_fn=prelu,scope='fc1')

        net=slim.flatten(net)
        cls_prob=slim.fully_connected(net,1,normalizer_fn=None,biases_initializer=tf.zeros_initializer(),activation_fn=tf.sigmoid,scope='fc2')
        bbox_pred=slim.fully_connected(net,4,normalizer_fn=None,biases_initializer=tf.zeros_initializer(),activation_fn=None,scope='fc3')

        cls_prob=tf.squeeze(cls_prob, [1], name='cls_prob')
        if training:
            cls_loss=cls_ohem(cls_prob,label)
            bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
            return cls_prob,bbox_pred,cls_loss,bbox_loss
        else:
            return cls_prob,bbox_pred


