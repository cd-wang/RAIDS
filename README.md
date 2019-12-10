## RAIDS
Security is of primary importance to vehicles. RAIDS employs a lightweight machine learning model to extract road contexts from sensory information (e.g., camera
images and distance sensor values) that are used to generate control signals for maneuvering the car. With such ongoing road context, RAIDS validates corresponding frames
observed on the in-vehicle network. Anomalous frames that substantially deviate from the road context will be discerned
as intrusions. We have implemented a prototype of RAIDS with neural networks, and conducted experiments on a Raspberry Pi with extensive datasets and meaningful intrusion
cases.

## Publication
Jingxuan Jiang, Chundong Wang, Sudipta Chattopadhyay, and Wei Zhang. *Road Context-aware Intrusion Detection System for Autonomous Cars*. In Proceedings of the 21st International Conference on Information and Communications Security (ICICS 2019). Beijing, China. 15-17 December 2019.

Paper link: <https://asset-group.github.io/papers/ICICS19-RAIDS.pdf>

## Environment setting up.

    For AWS Server //Ubuntu16.04 
    $ conda create -n raids python=3.6
    $ source activate raids
    $ conda install pytorch=0.4 -c Soumith
    $ conda install torchvision
    $ conda install matplotlib
    $ conda install pandas
    $ conda install cikit-image
    $ anaconda search -t conda tensorflow
    $ anaconda show anaconda/tensorflow  
    $ conda install -c anaconda tensorflow-gpu 1.12.0
    $ conda install keras 2.24
   
    
    For Raspberry Pi 2B+ //Raspbian// 64GB
    ///System setup
    Download Raspbian system: Raspbian Buster with desktop
    https://www.raspberrypi.org/downloads/raspbian/
    $ sudo apt-get install xrdp
    $ sudo apt-get install vnc4server tightvncserver
    /// environment
   
    Download tensorflow-1.12.0-cp35-none-linux_armv7l.whl
    https://github.com/lhelontra/tensorflow-on-arm/releases
    $ sudo apt install libatlas-base-dev
    $ pip3 install tensorflow
    
    $ sudo apt-get install libhdf5-serial-dev
    $ sudo pip3 install h5py
    $ sudo apt-get install python-scipy
    $ sudo pip3 install keras
    
    $ wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh
    $ sudo /bin/bash Miniconda3-latest-Linux-armv7l.sh
    $ conda config --add channels rpi
    $ conda install python=3.6
    $ pip3 install opencv-python
    $ conda install -y -c conda-forge opencv
    
 
## Dataset:
   
  1. Udacity Inc. The Udacity open source self-driving car project, April 2018. <https://github.com/udacity/self-driving-car>
  2. Udacity Inc. Udacity’s self-driving car simulator, July 2017. <https://github.com/udacity/self-driving-car-sim>
  3. Apollo.auto. Roadhackers platform in Baidu Apollo project, April 2018. <http://data.apollo.auto/static/pdf/road_hackers_en.pdf>
  4. Comma.ai. The Comma.ai driving dataset, October 2016. <https://github.com/commaai/research>
  5. Sully Chen. Sully Chen’s driving datasets (2017 & 2018), April 2018. <https://github.com/SullyChen/driving-datasets>
    
## Quick Start
    For example as commaai_data_try:
    First we need to train a CNN model to extract feature from each image. we use CAN information(steering angle) as label to train the models.The dataset is divided into training and testing with the training part being 70%.
    In feature_extraction_2cnn file, use python train.py to train the model and save the best model.
    $ python train.py
    Then try to use intrusion CAN model code modifies some of the values in the  csv dataset and also adds a column to it specifying if is an attack or not (1=attack,0=no attack)
    In try_commai_dark_data_2cnn/attack_csv file, use python attack_model_abs_a_random.py to attack CAN.
    $ python attack_model_abs_a_random.py
    In try_commai_dark_data_2cnn file, use python feature_extraction_intrusion_detection.py to train and test intrusion detection with contest.
    The dataset is divided into training and testing with the training part being 70%.
    $ python feature_extraction_intrusion_detection.py

