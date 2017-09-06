# rpi_self_driving
Self-Driving Car, using Raspberry Pi, Arduino Uno, and EasyVR

## Overview
In this project, I will use Basic DNN(Deep Neural Network) to recognize traffic sign images from camera module. The training/test dataset is from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Also, the code itself will not work properly unless you have a module to send a traffic image and receive result of a program. 

### Dependencies
#### Software:
  - Python 3.5
  - Tensorflow
  - Matplotlib
  - Numpy
  - Pandas
  - sklearn
#### Hardware:
  - Module to download the image (and send the result)

#### Recommended OS: 
  - Linux Ubuntu 16.04 LTS

### Running the code:
  - DNN Source Code: parameter.py
  - Begin Training: Train_Signs.py
  - Sign Classifier: Recognizer.py
  - Download dataset, and create a folder 'lab_2_data' under rpi_self_driving folder.
  
#### Actual Car in Motion
[YouTube Link #1 (3rd Person)](https://youtu.be/qZQRHKTS4-Y)\n
[YouTube Link #2 (1st Person)](https://youtu.be/GzHr-AEEOj4)
