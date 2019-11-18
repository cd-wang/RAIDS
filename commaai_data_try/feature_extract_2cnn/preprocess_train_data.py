from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
from skimage.exposure import rescale_intensity
from matplotlib.colors import rgb_to_hsv
from config import DataConfig
    
def make_hsv_data(path):
    df = pd.read_csv(path)
    num_rows = df.shape[0]
    
    X = np.zeros((num_rows, row, col, 3), dtype=np.uint8)
    for i in range(num_rows):
        if i % 1000 == 0:
            print( "Processed " + str(i) + " images...")

        path = df['fullpath'].iloc[i]
        img = load_img(data_path + path, target_size=(row, col))
        img = img_to_array(img)
        img = rgb_to_hsv(img)
        img = np.array(img, dtype=np.uint8)

        X[i] = img
        
    return X, np.array(df["angle"])

def make_color_data(path):
    df = pd.read_csv(path)
    num_rows = df.shape[0]
    
    X = np.zeros((num_rows, row, col, 3), dtype=np.uint8)
    for i in range(num_rows):
        if i % 1000 == 0:
            print( "Processed " + str(i) + " images...")

        path = df['fullpath'].iloc[i]
        img = load_img(data_path + path, target_size=(row, col))
        img = img_to_array(img)
        img = np.array(img, dtype=np.uint8)

        X[i] = img
        
    return X, np.array(df["angle"])

def make_grayscale_diff_data(path, num_channels=2):
    df = pd.read_csv(path)
    num_rows = df.shape[0]
    
    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print( "Processed " + str(i) + " images...")
        for j in range(num_channels):
            path0 = df['fullpath'].iloc[i - j - 1]
            path1 = df['fullpath'].iloc[i - j]
            img0 = load_img(data_path + path0, grayscale=True, target_size=(row, col))
            img1 = load_img(data_path + path1, grayscale=True, target_size=(row, col))
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            img = img1 - img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)

            X[i - num_channels, :, :, j] = img[:, :, 0]
    return X, np.array(df["angle"].iloc[num_channels:])

def make_grayscale_diff_tx_data(path, num_channels=2):
    df = pd.read_csv(path)
    num_rows = df.shape[0]
    
    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print( "Processed " + str(i) + " images...")
        path1 = df['fullpath'].iloc[i]
        img1 = load_img(data_path + path1, grayscale=True, target_size=(row, col))
        img1 = img_to_array(img1)
        for j in range(1, num_channels + 1):
            path0 = df['fullpath'].iloc[i - j]
            img0 = load_img(data_path + path0, grayscale=True, target_size=(row, col))
            img0 = img_to_array(img0)
            
            img = img1 - img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)

            X[i - num_channels, :, :, j - 1] = img[:, :, 0]
    return X, np.array(df["angle"].iloc[num_channels:])

def make_hsv_grayscale_diff_data(path, num_channels=2):
    df = pd.read_csv(path)
    # print("df: ", df)
    num_rows = df.shape[0]
    # print("num_rows: ", num_rows)
    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)

    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print( "Processed " + str(i) + " images...")
        for j in range(num_channels):
            # print("i: ", i)
            # print("j: ", j)
            path0 = df['filename'].iloc[i - j - 1]
            path1 = df['filename'].iloc[i - j]
            # print("path0: ", df['fullpath'])

            img0 = load_img(data_path + path0, target_size=(row, col))
            img1 = load_img(data_path + path1, target_size=(row, col))
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            img0 = rgb_to_hsv(img0)
            img1 = rgb_to_hsv(img1)

            # print("data_path + path0: ", data_path + path0)
            # print("data_path + path1: ", data_path + path1)

            img = img1[:, :, 2] - img0[:, :, 2]
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)
    
            X[i - num_channels, :, :, j] = img
            # print("num_channels: ", num_channels)
            # print("np.array(df[""].iloc[num_channels:]): ", df["angle"])

    return X, np.array(df["angle_convert"].iloc[num_channels:])
    
if __name__ == "__main__":
    config = DataConfig()
    # data_path = config.data_path
    row, col = config.img_height, config.img_width
    data_path ="/home/ubuntu/comma_day_data/02-02--10-16-58/"

    print( "Pre-processing phase 2 data...")
    X_train, y_train = make_hsv_grayscale_diff_data("{}/02-02--10-16-58_43700to247700f10.csv".format(data_path), 2)
    np.save("{}/X_train_round2_hsv_gray_diff_ch2_02-02--10-16-58".format(data_path), X_train)
    np.save("{}/y_train_round2_hsv_gray_diff_ch2_02-02--10-16-58".format(data_path), y_train)

