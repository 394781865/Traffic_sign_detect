import numpy as np
import argparse
import sys
import cv2
import os
from core.model import P_Net, R_Net, O_Net
from core.imdb import IMDB
from core.detector import Detector
from core.fcn_detector import FcnDetector
from core.MtcnnDetector import MtcnnDetector

import time
import shutil
from datetime import timedelta
import tensorflow as tf

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class DenseNet:
    def __init__(self, growth_rate, depth,
                 total_blocks, keep_prob,
                 model_type, dataset='GTSRB',
                 reduction=1.0,
                 bc_mode=False,
                 ):
        self.data_shape = (48,48,3)
        self.n_classes = 46
        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.model_type = model_type
        self.dataset_name = dataset
        self.batches_step = 0

        self._define_inputs()
        self._build_graph()
        self._initialize_session()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
        else:
            self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path


    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        #self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output

    def trainsition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier(
            [features_total, self.n_classes], name='W')
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block)
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes"):
            logits = self.trainsition_layer_to_classes(output)
        prediction = tf.nn.softmax(logits)
        #print(prediction.shape)
        self.prediction_out = tf.argmax(prediction, 1)

    def test(self, data):
        #print(data.shape)
        feed_dict = {
            self.images: data,
            self.is_training: False,
        }
            #fetches = [self.cross_entropy, self.accuracy]
        out = self.sess.run(self.prediction_out, feed_dict=feed_dict)
        #print(out.shape,out)
        return out
        #return mean_loss, mean_accuracy
def _measure_mean_and_std(images):
    # for every channel in image
    means = []
    stds = []
    # for every channel in image(assume this is last dimension)
    for ch in range(images.shape[-1]):
        means.append(np.mean(images[:, :, :, ch]))
        stds.append(np.std(images[:, :, :, ch]))
    return means,stds

def normalize_images(images):

    images = images.astype('float64')
    # for every channel in image(assume this is last dimension)
    images_means ,images_stds= _measure_mean_and_std(images)
    for i in range(images.shape[-1]):
        images[:, :, :, i] = ((images[:, :, :, i] - images_means[i]) / images_stds[i])
    return images

def visssss(img ,dets2,name , pred, thresh=0.998):
    print(pred)
    for i in range(dets2.shape[0]):
        #if dets2[i,:]==0:
        #   print('None')
        #   continue
        bbox = dets2[i, :4].astype('int32')
        score = dets2[i,4]
        if score > thresh and pred[i] < 45:
            clas = pred[i]
            print('up', clas)
            signname = collect[clas]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img,signname,(bbox[0]-3, bbox[1]-5),cv2.FONT_HERSHEY_SIMPLEX ,2,(0,255,0),2) 
            #cv2.rectangle(img, (bbox[1], bbox[3]), (bbox[0], bbox[2]), (255, 255, 0), 2)

        elif pred[i] < 45:
            clas = pred[i]
            print('down', clas)
            signname = collect[clas-1]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
            cv2.putText(img,signname,(bbox[0]-3, bbox[1]-5),cv2.FONT_HERSHEY_SIMPLEX ,2,(0,255,255),2) 
    ss = 'detect_%s'%(name)
    cv2.imwrite(ss,img)
def detectOneImg(prefix, epoch, batch_size, model, imgPath, test_mode="onet",
         thresh=[0.6, 0.6, 0.7], min_face_size=24,
         stride=2, slide_window=False, shuffle=False, vis=False):
    detectors = [None, None, None]
    #load densenet for classfication
    model.load_model()

    model_path=['%s-%s'%(x,y) for x,y in zip(prefix,epoch)]
    print(model_path)
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0],model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["rnet", "onet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "onet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    images_res = []
    img = cv2.imread(os.path.join(imgPath))
    boxes, boxes_c = mtcnn_detector.detect(img)
    #print(boxes_c.shape)
    #visssss(img, boxes_c, 'plain13134.jpg', thresh=0.998)
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4].astype('int32')
        if bbox[1]<0:
            bbox[1] = 0
        if bbox[0]<0:
            bbox[0] = 0
        if bbox[2]>2048:
            bbox[2] = 2048
        if bbox[3]>2048:
            bbox[3] = 2048
        crop = img[bbox[1]:bbox[3],bbox[0]:bbox[2], :]
        #print(boxes_c[i, :4])
        #print('crop:',crop.shape)
        crop = cv2.resize(crop, (48, 48))
        #cv2.imwrite("C:\\Users\\JINNIU\\Desktop\\liuzhen\\qinghua\\temp\\"+str(i)+'.jpg',crop)
        images_res.append(crop)
    images_res = np.array(images_res).astype(np.float32)
    images_res = normalize_images(images_res)
    pred = model.test(images_res)
    bg_box = np.where(pred==45)#if the class is 45, the image is the background and not the traffic sign. we omit these in the next step
    #print(pred)
    #print(len(boxes_cc))
    for ii in bg_box[0]:
        boxes_c[ii,:]=0
    #print(boxes_c)
    #del boxes_c[np.where(pred==45),:]
    visssss(img, boxes_c, imgPath , pred,thresh=0.998)


aa = "i2 i4 i5 il100 il60 il80 io ip p10 p11 p12 p19 p23 p26 p27 p3 p5 p6 pg ph4 ph4.5 ph5 pl100 pl120 pl20 pl30 pl40 pl5 pl50 pl60 pl70 pl80 pm20 pm30 pm55 pn pne po pr40 w13 w32 w55 w57 w59 wo"
collect = aa.split(' ')        
print(len(collect))
model = DenseNet(24,40,3,0.8,'DenseNet-BC',reduction=0.5,bc_mode=True)
imgPath = "79688.jpg"
detectOneImg(['./pnet/pnet','./rnet/rnet','./onet/onet'],[7,7,7],[1,1,1],model, imgPath)