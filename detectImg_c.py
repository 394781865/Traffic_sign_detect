import numpy as np
import argparse
import sys
import cv2
import os
from core.model import P_Net, R_Net, O_Net
from core.imdb import IMDB
#from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from core.MtcnnDetector import MtcnnDetector
# this scrip is for testing the detection bbox network.
def visssss(img ,dets2,name, thresh=0.998):
    for i in range(dets2.shape[0]):
        bbox = dets2[i, :4].astype('int32')
        score = dets2[i, 4]
        if score > thresh:

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            #cv2.rectangle(img, (bbox[1], bbox[3]), (bbox[0], bbox[2]), (255, 255, 0), 2)

        else:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
    ss = 'detect_%s'%(name)
    cv2.imwrite(os.path.join("C:\\Users\\JINNIU\\Desktop\\liuzhen\\qinghua\\out2",ss),img)
            
def test(prefix, epoch, batch_size, test_mode="onet",
         thresh=[0.6, 0.6, 0.7], min_face_size=24,
         stride=2, slide_window=False, shuffle=False, vis=False):
    #img = cv2.imread("./1111.jpg")
    detectors = [None, None, None]
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
    for name in os.listdir("C:\\Users\\JINNIU\\Desktop\\liuzhen\\qinghua"):
        img = cv2.imread(os.path.join("C:\\Users\\JINNIU\\Desktop\\liuzhen\\qinghua",name))
        boxes, boxes_c = mtcnn_detector.detect(img)
        visssss(img, boxes_c, name, thresh=0.998)
    #mtcnn_detector.vis_two(img,boxes, boxes_c,thresh=0.998)
test(['./pnet/pnet','./rnet/rnet','./onet/onet'],[7,7,7],[1,1,1])
#test(['./pnet/pnet','./rnet/rnet','./onet_hard/onet_hard'],[7,7,7],[1,1,1])

