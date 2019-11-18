class DataConfig(object):
    data_path = "/home/ubuntu/commaAI/all_data_ch2"
    data_name = "hsv_gray_diff_ch2"  # hsv_gray_diff_ch4
    img_height = 80
    img_width =  160
    num_channels = 2
    
class TrainConfig(DataConfig):
    model_name = "comma_large_dropout"
    batch_size = 100
    num_epoch = 1
    val_part = 6
    X_train_mean_path = "data/X_train_gray_diff2_mean.npy"
    
class TestConfig(TrainConfig):
    model_path = "/home/ubuntu/try_HMB/commaai_day_try/rambo_model_day/rambo_change_try2_resize_two_cnn/models/weights_hsv_gray_diff_comma_large_dropout-01-0.17489.hdf5"
    angle_train_mean = -0.004179079



