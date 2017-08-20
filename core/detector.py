import tensorflow as tf
import numpy as np

from config import config

class Detector(object):
    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph=tf.Graph()
        with graph.as_default():
            self.image_op=tf.placeholder(tf.float32,shape=[batch_size,data_size,data_size,3],name='input_image')
            self.cls_prob,self.bbox_pred=net_factory(self.image_op,training=False)
            self.sess=tf.Session(config=tf.ConfigProto( allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver=tf.train.Saver()
            saver.restore(self.sess,model_path)

        self.data_size = data_size
        self.batch_size = batch_size

    def predict(self, databatch):
        # access data
        # databatch: N x 3 x data_size x data_size
        scores = []
        batch_size = self.batch_size

        minibatch = []
        cur = 0
        n = databatch.shape[0]
        while cur < n:
            minibatch.append(databatch[cur:min(cur+batch_size, n), :, :, :])
            cur += batch_size
        cls_prob_list=[]
        bbox_pred_list=[]
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            if m < batch_size:
                keep_inds = np.arange(m)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m

            cls_prob,bbox_pred=self.sess.run([self.cls_prob,self.bbox_pred],feed_dict={self.image_op:data})
            cls_prob_list.append(cls_prob[:real_size])
            bbox_pred_list.append(bbox_pred[:real_size])

        return np.concatenate(cls_prob_list,axis=0),np.concatenate(bbox_pred_list,axis=0)
