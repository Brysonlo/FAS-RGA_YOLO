# FAS-YOLO&RGA-YOLO
A metal surface defect detection model FAS-YOLO based on modifying the loss function and lightweighting the head, combined with attention mechanism, and a metal surface defect detection model FAS-YOLO based on optimizing the multi-scale downsampling structure and lightweighting the backbone network
Setup prerequisites

**Setup**

This setup is for Windows 11 and NVIDIA-supported GPU.
1.	Install Python 3.11.4
2.	Updating GPU drives-Install NVIDIA drivers.
3.	CUDA installation- Cuda 10.2.89
4.	CuDnn Installation- Version 7.65
5.	Configure the environment according to the requirements. txt file in the project

**Data**
The dataset used in this article is an open-source dataset. For relevant download methods, please refer to the following two references:
1.	Song, K.; Yan, Y. A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects. Applied Surface Science 2013, 285, 858-864.
2.	Lv, X.; Duan, F.; Jiang, J.-j.; Fu, X.; Gan, L. Deep metallic surface defect detection: The new benchmark and detection network. Sensors 2020, 20, 1562.
After downloading the dataset and completing the dataset segmentation work, please arrange the dataset files according to the following structure：

Create test.txt and train.txt files for respective datasets that contain the list name of the images according to their directory. Then, create. names file that contains the list of classes in the dataset. Locate all files in the vedai/visdrone folder.
Create .data files for both datasets and locate in the ‘cfg’ folder. The file contains the following information:
```
dataset/
├── train/
│     ├──images/
│     │      ├── image1.jpg
│     │      ├── image2.jpg
│     │      └── image3.jpg
│     └── labels/
│            ├── lebel1.txt
│            ├── lebel2.txt
│            └── lebel3.txt
├── val/
│     ├──images/
│     │      ├── image1.jpg
│     │      ├── image2.jpg
│     │      └── image3.jpg
│     └── labels/
│            ├── lebel1.txt
│            ├── lebel2.txt
│            └── lebel3.txt
└── test/
      ├──images/
      │      ├── image1.jpg
      │      ├── image2.jpg
      │      └── image3.jpg
      └── labels/
             ├── lebel1.txt
             ├── lebel2.txt
             └── lebel3.txt
```
After completing the dataset creation, please create a new dataset yaml. file in the assets folder. The file template has been provided in this folder. Please note to label the location of the dataset file correctly.
```
path: datasets file location 
train: train/images  # train images
val: val/images  # val images
test: test/images   # test images
```

**Train&Valid**
1. Train on train.py
```
model = YOLO("Your model location") 
results = model.train(data="Your Dataset yaml file location",batch=,workers=,epochs=)
```
2.	Valid on val.py
```
model = YOLO(r'Your best model location')
```
