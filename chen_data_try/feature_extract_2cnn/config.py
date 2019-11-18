class DataConfig(object):
    data_path = "/home/ubuntu/chen_data/chen_all_ch2"
    data_name = "hsv_gray_diff_ch2"  # hsv_gray_diff_ch4
    img_height = 66
    img_width =  200
    num_channels = 2
    
class TrainConfig(DataConfig):
    model_name = "comma_large_dropout"
    batch_size = 100
    num_epoch = 1
    val_part = 6
    X_train_mean_path = "data/X_train_gray_diff2_mean.npy"
    
class TestConfig(TrainConfig):
    model_path = "models/weights_hsv_gray_diff_ch4_comma_large_dropout-01-0.00540.hdf5"
    angle_train_mean = -0.004179079



