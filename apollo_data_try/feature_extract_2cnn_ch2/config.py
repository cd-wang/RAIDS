class DataConfig(object):
    data_path = "/home/ubuntu/apollo_data_all"
    data_name = "hsv_gray_diff_ch2"  # hsv_gray_diff_ch4
    img_height = 100
    img_width =  100
    num_channels = 2
    
class TrainConfig(DataConfig):
    model_name = "comma_large_dropout"
    batch_size = 100
    num_epoch = 2
    val_part = 6
    X_train_mean_path = "data/X_train_gray_diff2_mean.npy"
    
class TestConfig(TrainConfig):
    model_path = "/home/ubuntu/apollo_data_all/models_apollo/model_try2_resize_2cnn/weights_hsv_gray_diff_comma_large_dropout-02-0.00057.hdf5"
    angle_train_mean = -0.004179079


class VisualizeConfig(object):
    pred_path = "submission/final.csv"
    true_path = "data/CH2_final_evaluation.csv"
    img_path = "phase2_test/center/*.jpg"
