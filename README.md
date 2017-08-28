# Traffic_sign_detect_tensorflow
This project is the implementation of the paper *<<Detection of Multiple Objects in Traffic Scenes
using Cascaded Convolutional Networks>>*" for traffic sign detection and recognition with tensorflow. 
The code is adapted from this [repo](https://github.com/zt706/tensorflow-mtcnn) and [repo](https://github.com/ikhlestov/vision_networks).
<br>
We use the [TsingHua-Tencent 100k](http://cg.cs.tsinghua.edu.cn/traffic-sign/)
traffic dataset as benchmark. Our project uses two-stage architecture just like fast-RCNN. The first step proposes the bounding boxes which contain traffic signs. This step adopts cascaded networks which compose of three sub-network(propose network, refine network, output network). The idea is similar to mtcnn which is designed for face detection, classification and alignment. In recognition step, the purpose is to put the cropped image from the bounding box into the network to get the predicted label of the cropped image. We reference the DenseNet to design our classification network.

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
<br>
`$ python runAll.py`
<br>
You also can train your dataset but you need to change some code for your purpose.
First you download the Tsinghua-Tencent dataset, you can use the file in prepare folder in [mtcnn](https://github.com/zt706/tensorflow-mtcnn) . After you prepare the dataset, you can run:
<br>
`$ python train_P_net.py`
`$ python train_R_net.py`
`$ python train_O_net.py`
<br>
in example folder to train the cascaded network.

In the end, you can run :
<br>
`$ python run_dense_net.py`
<br>
to train the classification network. You can find the run_dense_net.py in [densenet](https://github.com/ikhlestov/vision_networks).
However you have to adapt the files in data_providers folder to fit your dataset.<br>
After these steps, you can train your dataset for object detection and recognition.

## References
* [zt706/tensorflow-mtcnn](https://github.com/zt706/tensorflow-mtcnn)
* [ikhlestov/vision_networks](https://github.com/ikhlestov/vision_networks)
* [TsingHua-Tencent benchnark](http://cg.cs.tsinghua.edu.cn/traffic-sign/)




