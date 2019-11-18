class DataConfig(object):
    data_path = "/home/ubuntu/"
    data_name = "hsv_gray_diff_ch2"
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
    model_path = "models/"
    angle_train_mean = -0.004179079

