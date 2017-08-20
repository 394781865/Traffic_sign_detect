# Traffic_sign_detect
This project is used for traffic sign detection and recognition.
We use the qinghua_tencent traffic dataset as benchmark。 Our project uses two-stage architecture liking fast-rcnn. The first step proposes the bounding boxes which contain traffic sign. The proposing step adopts cascaded networks which compose of three sub_network(propose network, refine network, output network). The ideal is similar to mtcnn which is design for face detection and classification, alignment. In recognition step, the purpose is to put the crop image from the bounding box into the network to get the predicted label of the cropped image. We reference the DenseNet to design our classification network.

In the project, we have provided you with the trained model. You can run the code directly using:
$ python runAll.py

You also can train your dataset but you need to change some code.
