
Hardware£ºRaspberry Pi 2B+





Datasets:apollo_data,chen_data(2017,2018),commaai_data(day,night),udacity_data,udacity_simulate


For example as commaai_data_try:

First we need to train a CNN model to extract feature from each image. we use CAN information(steering angle) as label to train the models.The dataset is divided into training and testing with the training part being 70%.

In feature_extraction_2cnn file, use python train.py to train the model and save the best model.

Then try to use intrusion CAN model code modifies some of the values in the  csv dataset and also adds a column to it specifying if is an attack or not (1=attack,0=no attack)
In try_commai_dark_data_2cnn/attack_csv file, use python attack_model_abs_a_random.py to attack CAN.


In try_commai_dark_data_2cnn file, use python feature_extraction_intrusion_detection.py to train and test intrusion detection with contest.
The dataset is divided into training and testing with the training part being 70%.

In try_commai_dark_data_2cnn/detection_detection_without_context_dark file,
use python f_e_without_context.py to train and test intrusion detection  without contest.

The dataset is divided into training and testing with the training part being 70%.




