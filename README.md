# Traffic_sign_detect_tensorflow
This project is the implementation of the paper Detection of Multiple Objects in Traffic Scenes
using Cascaded Convolutional Networks for traffic sign detection and recognition with tensorflow. 
The code adapted from this [repo](https://github.com/zt706/tensorflow-mtcnn) and [repo](https://github.com/ikhlestov/vision_networks).
<br>
We use the [TsingHua-Tencent 100k](http://cg.cs.tsinghua.edu.cn/traffic-sign/)
taffic dataset as benchmark. Our project uses two-stage architecture liking fast-rcnn. The first step proposes the bounding boxes which contain traffic sign. The proposing step adopts cascaded networks which compose of three sub_network(propose network, refine network, output network). The ideal is similar to mtcnn which is design for face detection and classification, alignment. In recognition step, the purpose is to put the cropped image from the bounding box into the network to get the predicted label of the cropped image. We reference the DenseNet to design our classification network.

## Prerequisites
 * Python 3.5
 * TensorFlow >= 1.0.0
 * Numpy
 * easydict
 
## Result

 ![orig](https://github.com/ZhangDY1994/Traffic_sign_detect/blob/master/detect_79688.jpg)
 
 ![orig](https://github.com/ZhangDY1994/Traffic_sign_detect/blob/master/detect_82136.jpg)

## Usage
In the project, we have provided you with the trained model. You can run the code directly using:
`$ python runAll.py`

You also can train your dataset but you need to change some code for your purpose.
First you download the Tsinghua-Tencent dataset, you can use the file in prepare folder in [mtcnn](https://github.com/zt706/tensorflow-mtcnn) . After you prepare the dataset, you can run::
`$ python train_P_net.py`
`$ python train_R_net.py`
`$ python train_O_net.py`
in example folder to train the cascaded network.

In the end, you can run :
`python run_dense_net.py`
to train the classification network. You can find the run_dense_net.py in [densenet](https://github.com/ikhlestov/vision_networks)
But you adopt the data_providers for you own purpose.
After the step, you can train your dataset you object detection and recognition

## References
* [zt706/tensorflow-mtcnn](https://github.com/zt706/tensorflow-mtcnn)
* [ikhlestov/vision_networks](https://github.com/ikhlestov/vision_networks)
* [TsingHua-Tencent benchnark](http://cg.cs.tsinghua.edu.cn/traffic-sign/)




