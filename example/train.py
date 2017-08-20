#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import datetime
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '6')
sys.path.insert(0, "/home/zhangboyu/tensorflow/_python_build")
import tensorflow as tf

from core.loader import ImageLoader
from core.imdb import IMDB
from config import config

def train_model(base_lr,loss,data_num):
    lr_factor=0.1
    global_step = tf.Variable(0, trainable=False)
    boundaries=[int(epoch*data_num/config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    lr_values=[base_lr*(lr_factor**x) for x in range(0,len(config.LR_EPOCH)+1)]
    lr_op=tf.train.piecewise_constant(global_step, boundaries, lr_values)

    optimizer=tf.train.MomentumOptimizer(lr_op,0.9)
    train_op=optimizer.minimize(loss,global_step)
    return train_op,lr_op

def compute_accuracy(cls_prob,label):
    keep=(label>=0)
    pred=np.zeros_like(cls_prob)
    pred[cls_prob>0.5]=1
    return np.sum(pred[keep]==label[keep])*1.0/np.sum(keep)

def train_net(net_factory,prefix,end_epoch,imdb,
              net=12,frequent=50,base_lr=0.01):

    train_data=ImageLoader(imdb,net,config.BATCH_SIZE,shuffle=True)

    input_image=tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,net,net,3],name='input_image')
    label=tf.placeholder(tf.float32,shape=[config.BATCH_SIZE],name='label')
    bbox_target=tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,4],name='bbox_target')

    cls_prob_op,bbox_pred_op,cls_loss_op,bbox_loss_op=net_factory(input_image,label,bbox_target)

    train_op,lr_op=train_model(base_lr,cls_loss_op+bbox_loss_op,len(imdb))

    model_dir=prefix.rsplit('/',1)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    sess=tf.Session()
    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()
        accuracy_list=[]
        cls_loss_list=[]
        bbox_loss_list=[]
        for batch_idx,(image_x,(label_y,bbox_y))in enumerate(train_data):
            sess.run(train_op,feed_dict={input_image:image_x,label:label_y,bbox_target:bbox_y})
            if batch_idx%frequent==0:
                cls_pred,cls_loss,bbox_loss,lr=sess.run([cls_prob_op,cls_loss_op,bbox_loss_op,lr_op],feed_dict={input_image:image_x,label:label_y,bbox_target:bbox_y})
                accuracy=compute_accuracy(cls_pred,label_y)
                print "%s : Epoch: %d, Step: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,lr:%f "%(datetime.datetime.now(),cur_epoch,batch_idx,accuracy,cls_loss,bbox_loss,lr)
                accuracy_list.append(accuracy)
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)

        print "Epoch: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f "%(cur_epoch,np.mean(accuracy_list),np.mean(cls_loss_list),np.mean(bbox_loss_list))
        saver.save(sess,prefix,cur_epoch)

