# ------------------------------------------------
# My Main Project
  ## Github Rt ([Train](https://github.com/Wanita-8943/Main_Project#classification)) 
  ## Github Lt ([Train](https://github.com/Wanita-8943/My_Main_Project_Lt-#my_main_project_lt-)) 
# ------------------------------------------------


## EfficientNet-Keras

This repository contains Keras reimplementation of EfficientNet, the new convolutional neural network architecture from [EfficientNet](https://arxiv.org/abs/1905.11946) ([TensorFlow implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)). 

### Table of content
 1. [About EfficientNets](#about)
 2. [Examples](#examples)
 3. [Models](#models) 
 4. [Installation](#installation)


### About EfficientNet Models <a name="about"></a>

If you're new to EfficientNets, here is an explanation straight from the official TensorFlow implementation: 

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models. EfficientNets are based on AutoML and Compound Scaling. In particular, [AutoML Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) have been used to develop a mobile-size baseline network, named as EfficientNet-B0; Then, the compound scaling method is used to scale up this baseline to obtain EfficientNet-B1 to B7.

<table border="0">
<tr>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency:


* In high-accuracy regime, EfficientNet-B7 achieves state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet with 66M parameters and 37B FLOPS, being 8.4x smaller and 6.1x faster on CPU inference than previous best [Gpipe](https://arxiv.org/abs/1811.06965).

* In middle-accuracy regime, EfficientNet-B1 is 7.6x smaller and 5.7x faster on CPU inference than [ResNet-152](https://arxiv.org/abs/1512.03385), with similar ImageNet accuracy.

* Compared with the widely used [ResNet-50](https://arxiv.org/abs/1512.03385), EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% (+6.3%), under similar FLOPS constraint.


