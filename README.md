# MobileNetV3-SSD


MobileNetV3-SSD implementation in PyTorch 
Please move to the second version https://github.com/shaoshengsong/MobileNetV3-SSD-Compact-Version
have test results
If you want to try new technology, please go here https://github.com/shaoshengsong/quarkdet
A lightweight object detection includes multiple models
**Purpose**
Object Detection 
applied to object detection

environment

operating system: Ubuntu18.04

Python: 3.6

PyTorch: 1.1.0


**Object detection using MobileNetV3-SSD**

**Support Export ONNX**

Code Reference (Seriously refer to the following code)


**One SSD part**


[A PyTorch Implementation of Single Shot MultiBox Detector ](https://github.com/amdegroot/ssd.pytorch)

**Two MobileNetV3 parts**



[1 mobilenetv3 with pytorch，provide pre-train model](https://github.com/xiaolai-sqlai/mobilenetv3) 


[2 MobileNetV3 in pytorch and ImageNet pretrained models ](https://github.com/kuan-wang/pytorch-mobilenet-v3)


[3Implementing Searching for MobileNetV3 paper using Pytorch ](https://github.com/leaderj1001/MobileNetV3-Pytorch)


[4 MobileNetV1, MobileNetV2, VGG based SSD/SSD-lite implementation in Pytorch 1.0 / Pytorch 0.4. Out-of-box support for retraining on Open Images dataset. ONNX and Caffe2 support. Experiment Ideas like CoordConv. 
no discernible latency cost](https://github.com/qfgaohao/pytorch-ssd).


For 4, I have not done MobileNetV1, MobileNetV2, etc. code compatibility, only MobileNetV3 is available

**Download data**
This example uses cake and bread as an example, the reason is that the amount of data is small
The total size of all categories is 561G, cakes and breads are 3.2G

python3 open_images_downloader.py --root /media/santiago/a/data/open_images --class_names "Cake,Bread" --num_workers 20


**training process**

**first training**

python3 train_ssd.py --dataset_type open_images --datasets /media/santiago/data/open_images --net mb3-ssd-lite  --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 5


**Preload a previously trained model**

python3 train_ssd.py --dataset_type open_images --datasets /media/santiago/data/open_images --net mb3-ssd-lite --pretrained_ssd models/mb3-ssd-lite-Epoch-99-Loss-2.5194434596402613.pth  --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 200 --base_net_lr 0.001  --batch_size 5



**test an image**

python run_ssd_example.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-99-Loss-2.5194434596402613.pth models/open-images-model-labels.txt /home/santiago/picture/test.jpg

**Video detection**

python3 run_ssd_live_demo.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-99-Loss-2.5194434596402613.pth models/open-images-model-labels.txt


**Cake and Bread Pretrained model**


Link: https://pan.baidu.com/s/1byY1eJk3Hm3CTp-29KirxA 

Extraction code: qxwv 

**VOC Dataset Pretrained model**

Link: https://pan.baidu.com/s/1yt_IRY0RcgSxB-YwywoHuA 

Extraction code: 2sta 
